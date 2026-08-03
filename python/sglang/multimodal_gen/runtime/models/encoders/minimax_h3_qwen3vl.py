# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 HF Qwen3VL text/vision encoder.

Implements the tensor-encoding recipe:

- plain bf16 ``from_pretrained()`` (no device_map), then retain only the
    layers needed for layer-50 output; ``MINIMAX_H3_ENCODER_DEVICE`` selects
    CPU residency or the TP=8 Transformers tensor-parallel path
- cuDNN SDP is enabled during CUDA encode via
    ``torch.backends.cuda.enable_cudnn_sdp(True)``
- forward the multimodal backbone with all-ones attention_mask and
  mm_token_type_ids derived from ``config.image_token_id`` (position_ids
  omitted: model computes rope internally)
- return the unnormalized output after decoder layer 49; the encoder
  output contract is ``hidden_states[50]``, without retaining every
  intermediate hidden state or computing unused language-model logits

The encoder is pure-tensor (presentation building lives in
``pipelines_core/.../minimax_h3/presentation.py``). Supports persistent CPU
offload for single-GPU residency and recoverable destroy/reload offload for
Transformers TP residency.
"""

from __future__ import annotations

import gc
import os
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.sp_broadcast import minimax_h3_sp_ctx
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50
MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120
MINIMAX_H3_QWEN3VL_TRANSFORMERS_TP_SIZE = 8
MINIMAX_H3_ENCODER_DEVICE_ENV = "MINIMAX_H3_ENCODER_DEVICE"
MINIMAX_H3_ENCODER_TP_OFFLOAD_ENV = "MINIMAX_H3_ENCODER_TP_OFFLOAD"
MINIMAX_H3_QWEN3VL_TEXT_TP_PLAN = {
    "model.language_model.layers.*.self_attn.q_proj": "colwise",
    "model.language_model.layers.*.self_attn.k_proj": "colwise",
    "model.language_model.layers.*.self_attn.v_proj": "colwise",
    "model.language_model.layers.*.self_attn.o_proj": "rowwise",
    "model.language_model.layers.*.mlp.gate_proj": "colwise",
    "model.language_model.layers.*.mlp.up_proj": "colwise",
    "model.language_model.layers.*.mlp.down_proj": "rowwise",
}
MINIMAX_H3_QWEN3VL_VISION_TP_PLAN = {
    # qkv fused: gather output so existing reshape(seq, 3, num_heads, -1) still works
    "model.visual.blocks.*.attn.qkv": "colwise_gather_output",
    "model.visual.blocks.*.attn.proj": "rowwise_split_input",

    # vision MLP standard TP
    "model.visual.blocks.*.mlp.linear_fc1": "colwise",
    "model.visual.blocks.*.mlp.linear_fc2": "rowwise",

    # final merger / deepstack mergers
    "model.visual.merger.linear_fc1": "colwise",
    "model.visual.merger.linear_fc2": "rowwise",
    "model.visual.deepstack_merger_list.*.linear_fc1": "colwise",
    "model.visual.deepstack_merger_list.*.linear_fc2": "rowwise",
}
logger = init_logger(__name__)
MINIMAX_H3_QWEN3VL_TP_PLAN = {
    **MINIMAX_H3_QWEN3VL_TEXT_TP_PLAN,
    **MINIMAX_H3_QWEN3VL_VISION_TP_PLAN,
}


def _enable_cudnn_sdp(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(True)


def _tp_world_size() -> int:
    if not model_parallel_is_initialized():
        return 1
    return get_tp_world_size()


def _resolve_encoder_device_mode(*, sp_world: int, tp_world: int) -> bool:
    mode = os.getenv(MINIMAX_H3_ENCODER_DEVICE_ENV, "auto").strip().lower()
    if mode in ("", "auto"):
        return sp_world == 1 and tp_world == MINIMAX_H3_QWEN3VL_TRANSFORMERS_TP_SIZE
    if mode == "cpu":
        return False
    if mode in ("transformers_tp", "tp", "xpu_tp"):
        if sp_world != 1 or tp_world != MINIMAX_H3_QWEN3VL_TRANSFORMERS_TP_SIZE:
            raise ValueError(
                f"{MINIMAX_H3_ENCODER_DEVICE_ENV}={mode!r} requires SP=1 and "
                f"TP={MINIMAX_H3_QWEN3VL_TRANSFORMERS_TP_SIZE}, got "
                f"SP={sp_world}, TP={tp_world}."
            )
        return True
    raise ValueError(
        f"Unsupported {MINIMAX_H3_ENCODER_DEVICE_ENV}={mode!r}. Expected one "
        "of: auto, cpu, transformers_tp."
    )


def _rank_local_encoder_tp_device() -> torch.device:
    device = get_local_torch_device()
    if device.type == "cpu":
        raise RuntimeError(
            f"{MINIMAX_H3_ENCODER_DEVICE_ENV}=transformers_tp requires a rank-local "
            "accelerator device, but get_local_torch_device() returned cpu. "
            "Check that XPU is visible in this process and that "
            "SGLANG_DIFFUSION_PLATFORM_OVERRIDE is not set to cpu."
        )
    return device


def _should_destroy_tp_encoder_on_offload() -> bool:
    mode = os.getenv(MINIMAX_H3_ENCODER_TP_OFFLOAD_ENV, "destroy").strip().lower()
    if mode in ("", "destroy", "release", "reload", "true", "1", "yes"):
        return True
    if mode in ("keep", "none", "false", "0", "no"):
        return False
    raise ValueError(
        f"Unsupported {MINIMAX_H3_ENCODER_TP_OFFLOAD_ENV}={mode!r}. Expected "
        "one of: destroy, keep."
    )


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _retain_selected_lm_layer(model: torch.nn.Module, selected_layer: int) -> None:
    """Trim a Qwen3VL backbone to the requested pre-norm hidden-state index.

    The MiniMax H3 contract names the unnormalized output after the
    first ``N`` decoder layers ``hidden_states[N]``. It consumes layer 50 only,
    so keeping later layers, the final norm, every intermediate hidden state,
    and the causal-LM head is both unnecessary and prohibitively expensive for
    long Ref2VA presentations. This explicit definition also avoids depending
    on Transformers' version-specific ``output_hidden_states`` tuple indexing.
    """

    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise TypeError("Qwen3VL backbone must expose language_model.layers")
    original_num_layers = len(layers)
    if not 0 < selected_layer <= original_num_layers:
        raise ValueError(
            "selected Qwen3VL layer must be in "
            f"[1, {original_num_layers}], got {selected_layer}"
        )
    language_model.layers = layers[:selected_layer]
    language_model.norm = torch.nn.Identity()
    if hasattr(language_model, "config"):
        language_model.config.num_hidden_layers = selected_layer
    logger.info(
        "MiniMax H3 Qwen3VL retained %d/%d decoder layers for hidden_states[%d]",
        selected_layer,
        original_num_layers,
        selected_layer,
    )


class MiniMaxH3Qwen3VLHFEncoderStub:
    """Placeholder for non-main sequence-parallel ranks."""

    def __init__(self, *, hf_model_path: str) -> None:
        self.hf_model_path = hf_model_path

    def load_to_device(self) -> None:
        return

    def offload_to_cpu(self) -> None:
        return

    def encode_ids(self, *args, **kwargs):
        raise RuntimeError(
            "MiniMax H3 Qwen3VL encoder is stubbed on this sp rank; encode "
            "must run on the sp main rank"
        )


class MiniMaxH3Qwen3VLHFEncoder:
    """Frozen HF Qwen3VL-32B forward, layer-50 hidden-state extractor."""

    def __init__(
        self,
        *,
        hf_model_path: str,
        device: str = "cpu",
        use_transformers_tp: bool = False,
        tp_size: int = 1,
    ) -> None:
        self.selected_lm_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM
        self.hf_model_path = hf_model_path
        self.use_transformers_tp = use_transformers_tp
        self.tp_size = tp_size
        self.destroy_tp_encoder_on_offload = (
            use_transformers_tp and _should_destroy_tp_encoder_on_offload()
        )
        self.device = (
            _rank_local_encoder_tp_device()
            if use_transformers_tp
            else torch.device(device)
        )
        logger.info(
            "MiniMax H3 Qwen3VL encoder initialized on %s (TP=%d)",
            self.device,
            tp_size,
        )
        self.model: torch.nn.Module | None = None
        self.image_token_id: int | None = None
        self.video_token_id: int | None = None
        self._load_model()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        from transformers import Qwen3VLForConditionalGeneration

        load_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "trust_remote_code": False,
        }
        if self.use_transformers_tp:
            load_kwargs.update(
                {
                    "tp_plan": MINIMAX_H3_QWEN3VL_TP_PLAN,
                    "tp_size": self.tp_size,
                }
            )
            logger.info(
                "Loading MiniMax H3 Qwen3VL encoder with Transformers TP=%d on %s",
                self.tp_size,
                self.device,
            )
        else:
            logger.info(
                "Loading MiniMax H3 Qwen3VL encoder on CPU; enable TP=8/SP=1 "
                "to use the Transformers TP path."
            )
        causal_lm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.hf_model_path, **load_kwargs
        ).eval()
        self.image_token_id = int(causal_lm.config.image_token_id)
        self.video_token_id = int(causal_lm.config.video_token_id)
        # Consumers read hidden_states[50], never logits. Drop
        # the CausalLM wrapper/head and later decoder layers while everything
        # is still on CPU, before the first load_to_device().
        self.model = causal_lm.model
        _retain_selected_lm_layer(self.model, self.selected_lm_layer)
        del causal_lm

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("encoder is released; call load_to_device() before use")
        return self.model

    @classmethod
    def load_component(
        cls,
        *,
        component_model_path: str,
        component_name: str,
        config: dict[str, Any],
    ) -> MiniMaxH3Qwen3VLHFEncoder | MiniMaxH3Qwen3VLHFEncoderStub:
        """Load a flat Qwen3VL HF snapshot for a registered component."""
        if component_name != "text_encoder":
            raise ValueError(
                f"{cls.__name__} can only be used for component text_encoder, "
                f"got {component_name}"
            )
        config_path = os.path.join(component_model_path, "config.json")
        architectures = config.get("architectures")
        if not isinstance(architectures, list) or not any(
            isinstance(item, str) and item.startswith("Qwen3VL")
            for item in architectures
        ):
            raise ValueError(
                f"{config_path} must be a Qwen3VL HF snapshot config "
                f"(architectures=[Qwen3VL*...]), got architectures={architectures!r}"
            )

        sp_world, sp_rank = minimax_h3_sp_ctx()
        if sp_world > 1 and sp_rank != 0:
            logger.info(
                "MiniMax H3 Qwen3VL encoder stubbed on sp rank %d (main-rank only)",
                sp_rank,
            )
            return MiniMaxH3Qwen3VLHFEncoderStub(hf_model_path=component_model_path)
        tp_world = _tp_world_size()
        use_transformers_tp = _resolve_encoder_device_mode(
            sp_world=sp_world,
            tp_world=tp_world,
        )
        return cls(
            hf_model_path=component_model_path,
            use_transformers_tp=use_transformers_tp,
            tp_size=tp_world,
        )

    @torch.inference_mode()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a 1-D token stream; returns [seq_len, hidden_dim] bf16 CPU.

        pixel_values/image_grid_thw must be provided together for image
        presentations (fl2va) and omitted for text-only (t2va);
        pixel_values_videos/video_grid_thw likewise for video-frame
        presentations (ref2va video).
        """
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )
        if (
            not self.use_transformers_tp
            and next(self._require_model().parameters()).device.type != self.device.type
        ):
            raise RuntimeError(
                "encoder is offloaded; call load_to_device() before encode_ids()"
            )
        model = self._require_model()
        _enable_cudnn_sdp(self.device)
        ids = input_ids.to(self.device, torch.long)[None]
        kwargs: dict = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "output_hidden_states": False,
            "return_dict": True,
            "use_cache": False,
        }
        if pixel_values is not None or pixel_values_videos is not None:
            # get_rope_index routes grid consumption by token type:
            # {1: image_grid_thw, 2: video_grid_thw}. Marking video pads as 1
            # makes video blocks consume
            # image grid entries -> StopIteration.
            mm_types = torch.zeros_like(ids, dtype=torch.int32)
            mm_types[ids == self.image_token_id] = 1
            mm_types[ids == self.video_token_id] = 2
            kwargs["mm_token_type_ids"] = mm_types
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            kwargs["image_grid_thw"] = image_grid_thw.to(self.device, torch.long)
        if pixel_values_videos is not None:
            kwargs["pixel_values_videos"] = pixel_values_videos.to(
                self.device, torch.bfloat16
            )
            kwargs["video_grid_thw"] = video_grid_thw.to(self.device, torch.long)
        outputs = model(**kwargs)
        hidden = outputs.last_hidden_state[0].to(torch.bfloat16)
        if list(hidden.shape) != [int(ids.shape[1]), self.hidden_dim]:
            raise ValueError(
                f"unexpected hidden shape {list(hidden.shape)} for "
                f"seq_len={int(ids.shape[1])}"
            )
        return hidden.cpu()

    def offload_to_cpu(self) -> None:
        """Release accelerator-resident weights after an encode pass.

        Non-TP mode keeps the original persistent CPU residency. Transformers
        TP creates rank-local sharded state that is not reliably CPU-movable, so
        the memory-mode path releases it by deleting the module; load_to_device()
        restores it from the checkpoint for the next request.
        """
        if self.use_transformers_tp:
            if self.destroy_tp_encoder_on_offload and self.model is not None:
                logger.info(
                    "Releasing MiniMax H3 Qwen3VL TP encoder on %s; it will be "
                    "reloaded on the next text-encoding use",
                    self.device,
                )
                del self.model
                self.model = None
                gc.collect()
                _empty_device_cache(self.device)
            return
        self._require_model().to("cpu")
        _empty_device_cache(self.device)

    def load_to_device(self) -> None:
        if self.use_transformers_tp:
            self._load_model()
            return
        self._require_model().to(self.device)


EntryClass = MiniMaxH3Qwen3VLHFEncoder

__all__ = [
    "MiniMaxH3Qwen3VLHFEncoder",
    "MiniMaxH3Qwen3VLHFEncoderStub",
]
