"""Uniform-state (renoising) block-diffusion algorithm for DiffusionGemma.

Two phases, distinguished by `forward_batch.dllm_is_encoder`:
  * encoder/prefill: one forward to write the context KV cache (no denoising).
  * decoder/denoise: for one canvas, run `max_denoising_steps` reverse steps,
    feeding the previous step's logits back as self-conditioning.

Ported from the HF `diffusion_gemma` EntropyBoundSampler + linear temperature
schedule + StableAndConfident stopping. Defaults follow the checkpoint's
generation_config.json and may be overridden via --dllm-algorithm-config.
"""

import os
from typing import List, Tuple, Union

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

# Opt-in localization for device-side ``index out of bounds`` asserts (XPU/CUDA).
# These asserts fire asynchronously, so by default the error only surfaces at the
# next host<->device sync (e.g. a ``.item()`` call), pointing at the wrong line.
# Set ``SGLANG_DLLM_DEBUG_SYNC=1`` to synchronize after each indexing op and
# validate canvas token ids eagerly, so the real culprit is reported in place.
_DLLM_DEBUG_SYNC = os.environ.get("SGLANG_DLLM_DEBUG_SYNC", "0") == "1"


class Gemma4Renoise(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.canvas_length = config.block_size

        ac = config.algorithm_config or {}
        self.max_denoising_steps = ac.get("max_denoising_steps", 48)
        if self.max_denoising_steps < 1:
            raise ValueError("max_denoising_steps must be >= 1")
        s = ac.get("sampler_config", {})
        self.entropy_bound = s.get("entropy_bound", 0.1)
        t = ac.get("temperature_schedule", {})
        self.t_min = t.get("t_min", 0.4)
        self.t_max = t.get("t_max", 0.8)
        st = ac.get("stopping_config", {})
        self.confidence_threshold = st.get("confidence_threshold", 0.005)
        self.stability_threshold = st.get("stability_threshold", 1)
        self._accepted_mask = None

        self.vocab_size = None
        self._base_seed = ac.get("seed", None)
        self._generator = None

    def _temperature(self, step: int) -> float:
        return self.t_min + (self.t_max - self.t_min) * (
            step / self.max_denoising_steps
        )

    def _seed_generator(self, device) -> torch.Generator:
        # One generator drives the whole decode so every canvas draw is reproducible
        # (when `seed` is set) and, under TP, identical across ranks: each rank samples
        # the canvas from all-gathered logits, so divergent draws would corrupt
        # attention. Re-seed per decode and broadcast rank 0's seed so ranks agree.
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
        if self._base_seed is not None:
            seed = int(self._base_seed)
        else:
            seed = int(torch.randint(0, torch.iinfo(torch.int32).max, ()).item())
        if get_tensor_model_parallel_world_size() > 1:
            t = torch.tensor(
                [seed if get_tensor_model_parallel_rank() == 0 else 0],
                dtype=torch.int64,
                device=device,
            )
            seed = int(tensor_model_parallel_all_reduce(t).item())
        self._generator.manual_seed(seed)
        return self._generator

    def _init_canvas(self, shape, device, gen) -> torch.Tensor:
        return torch.randint(
            low=0, high=self.vocab_size, size=shape, device=device, generator=gen
        )

    def _sync(self, device) -> None:
        # Force a host<->device sync so an async device-side assert (e.g. the XPU
        # ``index out of bounds`` in the embedding gather) is raised at this point
        # instead of leaking to an unrelated later line. No-op unless debugging.
        if not _DLLM_DEBUG_SYNC:
            return
        mod = getattr(torch, device.type, None)
        if mod is not None and hasattr(mod, "synchronize"):
            mod.synchronize()

    def _write_canvas(self, forward_batch, canvas) -> None:
        # The canvas tokens become ``input_ids`` for the next forward, where they
        # index ``embed_tokens`` (an ``nn.Embedding`` of ``vocab_size`` rows). A
        # stray id outside ``[0, vocab_size)`` triggers a device-side out-of-bounds
        # assert on XPU (and silent garbage reads on CUDA). ``torch.multinomial``
        # can emit such ids when fed NaN/Inf or all-zero rows, so clamp defensively
        # before handing the canvas back to the model.
        ids = canvas.reshape(-1)
        if _DLLM_DEBUG_SYNC:
            lo = int(ids.min().item())
            hi = int(ids.max().item())
            assert 0 <= lo and hi < self.vocab_size, (
                f"canvas token id out of range: [{lo}, {hi}] not within "
                f"[0, {self.vocab_size})"
            )
        ids = ids.clamp_(0, self.vocab_size - 1)
        forward_batch.input_ids[:] = ids
        self._sync(forward_batch.input_ids.device)

    def _accept(self, current, denoiser, token_entropy) -> torch.Tensor:
        # EntropyBound: accept the lowest-entropy positions within entropy_bound.
        # Store the mask so _renoise re-noises exactly its complement.
        sorted_e, idx = torch.sort(token_entropy, dim=-1, descending=False)
        cum = torch.cumsum(sorted_e, dim=-1)
        sel = (cum - sorted_e) <= self.entropy_bound
        self._accepted_mask = torch.zeros_like(sel).scatter(-1, idx, sel)
        return torch.where(self._accepted_mask, denoiser, current)

    def _renoise(self, accepted, gen) -> torch.Tensor:
        # Re-noise every non-accepted position with a fresh uniform token.
        rand = self._init_canvas(accepted.shape, accepted.device, gen)
        return torch.where(self._accepted_mask, accepted, rand)

    def _stop(self, history, argmax, token_entropy) -> torch.Tensor:
        # Per-request [bs] bool: each request stops on its own stability/confidence.
        bs = argmax.shape[0]
        if len(history) == self.stability_threshold:
            stable = torch.ones(bs, dtype=torch.bool, device=argmax.device)
            for c in history:
                stable &= (argmax == c).all(dim=1)
        else:
            stable = torch.zeros(bs, dtype=torch.bool, device=argmax.device)
        history.append(argmax)
        if len(history) > self.stability_threshold:
            history.pop(0)
        confident = token_entropy.mean(dim=1) < self.confidence_threshold
        return stable & confident

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self.vocab_size is None:
            self.vocab_size = model_runner.model_config.hf_config.text_config.vocab_size

        if forward_batch.dllm_is_encoder:
            # Skip an empty (fully-cached) encode round, rope can't reshape 0 tokens.
            if forward_batch.input_ids.numel() == 0:
                return None, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        bs = forward_batch.batch_size
        L = self.canvas_length
        device = forward_batch.input_ids.device
        gen = self._seed_generator(device)

        current = self._init_canvas((bs, L), device, gen)
        self._write_canvas(forward_batch, current)
        self_cond = None
        history: List[torch.Tensor] = []
        out = None
        # Each request freezes its canvas once it stops while the rest keep denoising.
        done = torch.zeros(bs, dtype=torch.bool, device=device)
        # Emit the greedy argmax (not the multinomial canvas), frozen per-item on stop.
        argmax = current.clone()

        for step in reversed(range(1, self.max_denoising_steps + 1)):
            forward_batch.dllm_self_conditioning_logits = self_cond
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            # The embedding gather inside the forward is where a stray canvas id
            # trips the device-side assert; sync here so it localizes correctly.
            self._sync(device)

            logits = (out.logits_output.full_logits / self._temperature(step)).view(
                bs, L, -1
            )
            token_entropy = torch.distributions.Categorical(logits=logits).entropy()
            probs = torch.softmax(logits, dim=-1)
            # ``softcapping``/temperature scaling and masked positions can leave
            # NaN/Inf in ``logits`` and hence non-finite or all-zero ``probs`` rows.
            # ``torch.multinomial`` then returns out-of-range ids (observed as an
            # XPU ``index out of bounds`` assert on the next embedding gather), so
            # scrub the distribution to a valid one before sampling.
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            invalid = probs.sum(dim=-1, keepdim=True) <= 0
            if bool(invalid.any()):
                probs = torch.where(invalid, torch.ones_like(probs), probs)
            denoiser = torch.multinomial(
                probs.reshape(bs * L, -1), num_samples=1, generator=gen
            ).view(bs, L)

            keep = done.view(bs, 1)
            argmax = torch.where(keep, argmax, torch.argmax(logits, dim=-1))
            accepted = self._accept(current, denoiser, token_entropy)
            current = torch.where(keep, current, self._renoise(accepted, gen))
            self._write_canvas(forward_batch, current)
            if self_cond is None:
                self_cond = logits.reshape(bs * L, -1)
            else:
                self_cond = torch.where(
                    done.view(bs, 1, 1), self_cond.view(bs, L, -1), logits
                ).reshape(bs * L, -1)

            newly = self._stop(history, argmax, token_entropy) & (~done)
            if bool(newly.any()):
                done |= newly
            if bool(done.all()):
                break

        next_token_ids = [argmax[i] for i in range(bs)]
        return out.logits_output, next_token_ids, out.can_run_graph


Algorithm = Gemma4Renoise
