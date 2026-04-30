import itertools
import unittest

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm, rms_normalize_triton


class TestRMSNormalizeTriton(unittest.TestCase):
    def test_rms_normalize_without_weight(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 4.0]])
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        out = rms_normalize_triton(x, 1e-6)

        self.assertIs(out, x)
        self.assertTrue(torch.allclose(out, expected))

    def test_rms_normalize_with_weight(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 4.0]])
        weight = torch.tensor([1.0, 0.5, 2.0])
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight

        out = rms_normalize_triton(x, 1e-6, weight)

        self.assertIs(out, x)
        self.assertTrue(torch.allclose(out, expected))

    def test_rms_normalize_non_contiguous(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t()
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        out = rms_normalize_triton(x, 1e-6)

        self.assertIs(out, x)
        self.assertTrue(torch.allclose(out, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_rms_normalize_cuda_in_place(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        data_ptr = x.data_ptr()
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        out = rms_normalize_triton(x, 1e-6)

        self.assertIs(out, x)
        self.assertEqual(out.data_ptr(), data_ptr)
        self.assertTrue(torch.allclose(out, expected))


class TestRMSNorm(unittest.TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(self, num_tokens, hidden_size, add_residual, dtype, seed):
        torch.manual_seed(seed)

        layer = RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2))

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_rms_norm_test(*params)


class TestGemmaRMSNorm(unittest.TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_gemma_rms_norm_test(
        self, num_tokens, hidden_size, add_residual, dtype, seed
    ):
        torch.manual_seed(seed)

        layer = GemmaRMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-3, rtol=1e-3))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-3, rtol=1e-3))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3))

    def test_gemma_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_gemma_rms_norm_test(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
