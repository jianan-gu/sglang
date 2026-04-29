"""
Tests for the PyTorch native implementations that replace Triton kernels
in the DeepSeek V4 branch (blzheng/sglang:beilei/deepseek_v4).

These tests verify mathematical correctness of:
1. act_quant_pytorch - FP8 block-wise quantization
2. fused_scale_torch - Fused scale operation
3. topk_transform_512_pytorch_vectorized - Top-k with page table translation
4. _init_compressed_attn_metadata_pytorch - Compressed attention metadata
5. apply_rotary_emb_triton (PyTorch native) - Rotary embedding
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ============================================================
# 1. act_quant_pytorch
# ============================================================

def act_quant_pytorch(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0

    fp8_max = 448.0
    fp8_min = -448.0

    orig_shape = x.shape
    N = x.size(-1)
    x_flat = x.view(-1, N).float()
    M = x_flat.size(0)
    num_groups = N // block_size

    x_blocked = x_flat.view(M, num_groups, block_size)
    amax = x_blocked.abs().amax(dim=2)
    amax = amax.clamp(min=1e-4)

    round_scale = scale_fmt is not None
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
    else:
        scale = amax / fp8_max

    y = x_blocked / scale.unsqueeze(2)
    y = y.clamp(fp8_min, fp8_max)

    y = y.view(orig_shape).to(torch.float8_e4m3fn)
    s = scale.view(*orig_shape[:-1], num_groups)

    return y, s


def test_act_quant_basic():
    """Test basic FP8 quantization with known values."""
    torch.manual_seed(42)
    x = torch.randn(2, 256)  # 2 rows, 256 cols, block_size=128 -> 2 groups

    y, s = act_quant_pytorch(x, block_size=128)

    assert y.dtype == torch.float8_e4m3fn
    assert y.shape == x.shape
    assert s.dtype == torch.float32
    assert s.shape == (2, 2)  # 2 rows, 2 groups

    # Dequantize and check it's close to original
    y_float = y.float()
    y_dequant = y_float.view(2, 2, 128) * s.unsqueeze(-1)
    y_dequant = y_dequant.view(2, 256)

    # FP8 e4m3 has limited precision, but should be reasonably close
    max_err = (x - y_dequant).abs().max().item()
    print(f"  act_quant basic: max dequant error = {max_err:.6f}")
    assert max_err < 0.1, f"Dequantization error too large: {max_err}"


def test_act_quant_scale_range():
    """Test that quantized values are within FP8 range."""
    torch.manual_seed(42)
    x = torch.randn(4, 3, 128) * 100  # Large values

    y, s = act_quant_pytorch(x, block_size=128)

    # All quantized values should be within [-448, 448]
    y_float = y.float()
    assert y_float.abs().max().item() <= 448.0
    assert s.shape == (4, 3, 1)


def test_act_quant_with_scale_fmt():
    """Test quantization with power-of-2 scale rounding."""
    torch.manual_seed(42)
    x = torch.randn(2, 128) * 10

    y, s = act_quant_pytorch(x, block_size=128, scale_fmt="e8m0")

    # Scales should be powers of 2
    log2_scales = torch.log2(s)
    assert torch.allclose(log2_scales, log2_scales.round(), atol=1e-5), \
        "Scales should be powers of 2 when scale_fmt is set"


def test_act_quant_multidim():
    """Test with various tensor shapes."""
    torch.manual_seed(42)
    for shape in [(1, 128), (8, 256), (2, 4, 384), (3, 2, 512)]:
        x = torch.randn(*shape)
        y, s = act_quant_pytorch(x, block_size=128)
        assert y.shape == x.shape
        num_groups = shape[-1] // 128
        expected_scale_shape = (*shape[:-1], num_groups)
        assert s.shape == expected_scale_shape, f"Shape {shape}: expected scale {expected_scale_shape}, got {s.shape}"


# ============================================================
# 2. fused_scale_torch
# ============================================================

def fused_scale_torch(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    acc = weight.reshape(-1).float() * out_scale * q_scale.reshape(-1).float()
    out = acc.to(out_dtype).reshape(B, H, 1)
    return out


def test_fused_scale_basic():
    """Test fused scale with simple values."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    q_scale = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    out_scale = 2.0

    result = fused_scale_torch(weight, out_scale, q_scale)

    expected = torch.tensor([[[1.0], [2.0]], [[6.0], [8.0]]])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    assert result.shape == (2, 2, 1)


def test_fused_scale_dtypes():
    """Test that output dtype is correctly promoted."""
    weight = torch.randn(4, 8, dtype=torch.float32)
    q_scale = torch.randn(4, 8, dtype=torch.float32)

    result = fused_scale_torch(weight, 1.5, q_scale)
    assert result.dtype == torch.float32
    assert result.shape == (4, 8, 1)

    # Manual reference
    expected = (weight.reshape(-1) * 1.5 * q_scale.reshape(-1)).reshape(4, 8, 1)
    assert torch.allclose(result, expected, atol=1e-6)


# ============================================================
# 3. topk_transform_512_pytorch_vectorized
# ============================================================

def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores[~valid_mask] = float("-inf")

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(masked_scores, k=actual_k, dim=1, largest=True, sorted=False)
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        padding = torch.zeros((batch_size, TOPK - actual_k), dtype=torch.int32, device=device)
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, TOPK)
    gathered_scores = scores[batch_indices.flatten(), raw_indices.clamp(min=0).flatten()].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    if needs_sequential.any():
        sequential_indices = torch.arange(TOPK, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
        sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

        raw_indices = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK),
            torch.where(sequential_valid, sequential_indices, torch.tensor(-1, device=device, dtype=torch.int32)),
            raw_indices,
        )
        valid_topk = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK), sequential_valid, valid_topk
        )

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices = torch.where(valid_topk, page_indices, torch.tensor(-1, device=device, dtype=torch.int32))

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = torch.where(valid_topk, raw_indices, torch.tensor(-1, device=device, dtype=torch.int32))
        out_raw_indices.copy_(raw_indices)


def test_topk_small_seq():
    """Test topk when seq_len <= TOPK (should return sequential indices)."""
    batch_size = 2
    page_size = 64
    seq_lens = torch.tensor([100, 200], dtype=torch.int32)
    max_seq_len = 200
    max_pages = (max_seq_len + page_size - 1) // page_size

    scores = torch.randn(batch_size, max_seq_len)
    # Create simple page table: identity mapping
    page_tables = torch.arange(max_pages).unsqueeze(0).expand(batch_size, -1).contiguous().to(torch.int64)

    out_page_indices = torch.empty(batch_size, 512, dtype=torch.int32)
    out_raw_indices = torch.empty(batch_size, 512, dtype=torch.int32)

    topk_transform_512_pytorch_vectorized(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )

    # For seq_len <= 512, should return sequential indices
    for i in range(batch_size):
        sl = seq_lens[i].item()
        valid_raw = out_raw_indices[i, :sl]
        # Should be 0, 1, 2, ..., sl-1
        expected = torch.arange(sl, dtype=torch.int32)
        assert torch.equal(valid_raw, expected), f"Batch {i}: expected sequential indices"
        # Invalid entries should be -1
        assert (out_raw_indices[i, sl:] == -1).all(), f"Batch {i}: padding should be -1"


def test_topk_large_seq():
    """Test topk when seq_len > TOPK (should select top-k scores)."""
    batch_size = 1
    page_size = 64
    seq_len = 1024
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    max_pages = (seq_len + page_size - 1) // page_size

    # Create scores where highest values are at known positions
    scores = torch.zeros(batch_size, seq_len)
    top_positions = torch.arange(512)  # positions 0-511 get highest scores
    scores[0, top_positions] = torch.arange(512, dtype=torch.float32) + 1000

    page_tables = torch.arange(max_pages).unsqueeze(0).contiguous().to(torch.int64)
    out_page_indices = torch.empty(batch_size, 512, dtype=torch.int32)
    out_raw_indices = torch.empty(batch_size, 512, dtype=torch.int32)

    topk_transform_512_pytorch_vectorized(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )

    # All selected raw indices should be in [0, 511]
    valid = out_raw_indices[0] >= 0
    selected = out_raw_indices[0][valid]
    assert selected.numel() == 512
    selected_sorted = selected.sort().values
    assert torch.equal(selected_sorted, torch.arange(512, dtype=torch.int32))


def test_topk_page_translation():
    """Test that page table translation is correct."""
    batch_size = 1
    page_size = 4  # Small page size for easy testing
    seq_len = 8
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    max_pages = 2  # 8 / 4 = 2 pages

    scores = torch.randn(batch_size, seq_len)
    # Page table: physical page 0 -> logical 5, physical page 1 -> logical 10
    page_tables = torch.tensor([[5, 10]], dtype=torch.int64)

    out_page_indices = torch.empty(batch_size, 512, dtype=torch.int32)
    out_raw_indices = torch.empty(batch_size, 512, dtype=torch.int32)

    topk_transform_512_pytorch_vectorized(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )

    # seq_len=8 <= 512, so sequential indices 0-7 should be used
    page_bits = 2  # page_size=4, bits=2
    for j in range(seq_len):
        raw_idx = out_raw_indices[0, j].item()
        page_idx = raw_idx >> page_bits
        offset = raw_idx & (page_size - 1)
        physical_page = page_tables[0, page_idx].item()
        expected = (physical_page << page_bits) | offset
        actual = out_page_indices[0, j].item()
        assert actual == expected, f"Index {j}: expected {expected}, got {actual}"


# ============================================================
# 4. _init_compressed_attn_metadata_pytorch
# ============================================================

def _init_compressed_attn_metadata_pytorch(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
]:
    bs = seq_lens.shape[0]
    device = seq_lens.device

    c4_should_compress = (seq_lens % 4) == 0
    c4_out_loc = torch.where(c4_should_compress, raw_out_loc // 4, torch.zeros_like(raw_out_loc))
    c4_positions = (positions & (~3)).to(torch.int32)
    c4_seq_lens_raw = (seq_lens // 4).to(torch.int32)
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1)

    c128_should_compress = (seq_lens % 128) == 0
    c128_out_loc = torch.where(c128_should_compress, raw_out_loc // 128, torch.zeros_like(raw_out_loc))
    c128_positions = (positions & (~127)).to(torch.int32)
    c128_seq_lens_raw = (seq_lens // 128).to(torch.int32)
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1)

    c128_page_indices: Optional[torch.Tensor] = None
    if compute_page_indices:
        assert page_table is not None
        assert page_size > 0

        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages

        offsets = torch.arange(c128_max_seq_len, device=device, dtype=torch.int32)
        page_idx = offsets // c128_page_size
        offset_in_page = offsets % c128_page_size

        page_idx_clamped = torch.clamp(page_idx, max=max_pages - 1)
        page_idx_expanded = page_idx_clamped.unsqueeze(0).expand(bs, -1)
        page_table_vals = torch.gather(page_table, dim=1, index=page_idx_expanded.to(torch.int64)).to(torch.int32)

        c128_page_indices_vals = page_table_vals * c128_page_size + offset_in_page.unsqueeze(0)
        valid_mask = offsets.unsqueeze(0) < c128_seq_lens_raw.unsqueeze(1)
        c128_page_indices = torch.where(
            valid_mask, c128_page_indices_vals, torch.tensor(-1, dtype=torch.int32, device=device)
        )

    return (
        c4_out_loc.to(torch.int32), c4_positions, c4_seq_lens_raw,
        c4_seq_lens_clamp1, c128_out_loc.to(torch.int32), c128_positions,
        c128_seq_lens_clamp1, c128_page_indices,
    )


def test_compressed_metadata_c4():
    """Test compress-by-4 metadata computation."""
    seq_lens = torch.tensor([16, 12, 3], dtype=torch.int32)
    positions = torch.tensor([15, 11, 2], dtype=torch.int32)
    raw_out_loc = torch.tensor([100, 200, 300], dtype=torch.int32)

    result = _init_compressed_attn_metadata_pytorch(
        seq_lens, positions, raw_out_loc, compute_page_indices=False
    )
    c4_out_loc, c4_positions, c4_seq_lens_raw, c4_seq_lens_clamp1 = result[:4]

    # seq_lens % 4 == 0 for first two
    assert c4_out_loc[0].item() == 100 // 4  # 25
    assert c4_out_loc[1].item() == 200 // 4  # 50
    assert c4_out_loc[2].item() == 0  # 3 % 4 != 0, so 0

    # positions & (~3) = align down to multiple of 4
    assert c4_positions[0].item() == 12  # 15 & ~3 = 12
    assert c4_positions[1].item() == 8   # 11 & ~3 = 8
    assert c4_positions[2].item() == 0   # 2 & ~3 = 0

    # seq_lens // 4
    assert c4_seq_lens_raw[0].item() == 4
    assert c4_seq_lens_raw[1].item() == 3
    assert c4_seq_lens_raw[2].item() == 0

    # clamp to min 1
    assert c4_seq_lens_clamp1[2].item() == 1


def test_compressed_metadata_c128():
    """Test compress-by-128 metadata computation."""
    seq_lens = torch.tensor([256, 128, 100], dtype=torch.int32)
    positions = torch.tensor([255, 127, 99], dtype=torch.int32)
    raw_out_loc = torch.tensor([1280, 640, 500], dtype=torch.int32)

    result = _init_compressed_attn_metadata_pytorch(
        seq_lens, positions, raw_out_loc, compute_page_indices=False
    )
    c128_out_loc, c128_positions, c128_seq_lens_clamp1 = result[4], result[5], result[6]

    assert c128_out_loc[0].item() == 1280 // 128  # 10
    assert c128_out_loc[1].item() == 640 // 128   # 5
    assert c128_out_loc[2].item() == 0  # 100 % 128 != 0

    assert c128_positions[0].item() == 128  # 255 & ~127 = 128
    assert c128_positions[1].item() == 0    # 127 & ~127 = 0
    assert c128_positions[2].item() == 0    # 99 & ~127 = 0

    assert c128_seq_lens_clamp1[2].item() == 1  # clamped from 0


def test_compressed_metadata_page_indices():
    """Test page index computation."""
    bs = 2
    page_size = 256  # c128_page_size = 256/128 = 2
    max_pages = 4
    seq_lens = torch.tensor([512, 256], dtype=torch.int32)
    positions = torch.tensor([511, 255], dtype=torch.int32)
    raw_out_loc = torch.tensor([1000, 500], dtype=torch.int32)
    page_table = torch.tensor([[10, 20, 30, 40], [5, 15, 25, 35]], dtype=torch.int32)

    result = _init_compressed_attn_metadata_pytorch(
        seq_lens, positions, raw_out_loc, page_table, page_size, True
    )
    c128_page_indices = result[7]

    assert c128_page_indices is not None
    c128_page_size = page_size // 128  # 2
    c128_max_seq_len = c128_page_size * max_pages  # 8

    assert c128_page_indices.shape == (bs, c128_max_seq_len)

    # Batch 0: seq_len=512, c128_seq_len = 512//128 = 4
    # Offsets 0,1 -> page_idx=0, physical=10, result = 10*2 + offset
    assert c128_page_indices[0, 0].item() == 10 * 2 + 0  # 20
    assert c128_page_indices[0, 1].item() == 10 * 2 + 1  # 21
    # Offsets 2,3 -> page_idx=1, physical=20
    assert c128_page_indices[0, 2].item() == 20 * 2 + 0  # 40
    assert c128_page_indices[0, 3].item() == 20 * 2 + 1  # 41
    # Beyond c128_seq_len=4 should be -1
    assert c128_page_indices[0, 4].item() == -1


# ============================================================
# 5. apply_rotary_emb_triton (PyTorch native version)
# ============================================================

def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb_reference(x, freqs_cis, positions=None, inverse=False):
    """Reference implementation using complex arithmetic."""
    y = x.clone()
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if positions is not None:
        fc = freqs_cis[positions]
    else:
        fc = freqs_cis
    if inverse:
        fc = fc.conj()
    if x_complex.ndim == 3:
        fc = fc.unsqueeze(1)
    x_rot = torch.view_as_real(x_complex * fc).flatten(-2)
    return x_rot.to(x.dtype)


def apply_rotary_emb_pytorch_native(x, freqs_cis, positions=None, inverse=False):
    """The PyTorch native implementation from the branch."""
    is_3d = x.ndim == 3

    if is_3d:
        batch_size, n_heads, rope_dim = x.shape
    else:
        batch_size, rope_dim = x.shape
        n_heads = 1

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)

    if positions is not None:
        assert positions.shape == (batch_size,)
        freqs_real = freqs_real[positions]
    else:
        assert freqs_real.shape[0] == batch_size

    x_real = x[..., 0::2].float()
    x_imag = x[..., 1::2].float()

    freq_r = freqs_real[..., 0::2]
    freq_i = freqs_real[..., 1::2]

    if is_3d:
        freq_r = freq_r.unsqueeze(1)
        freq_i = freq_i.unsqueeze(1)

    if inverse:
        out_real = x_real * freq_r + x_imag * freq_i
        out_imag = x_imag * freq_r - x_real * freq_i
    else:
        out_real = x_real * freq_r - x_imag * freq_i
        out_imag = x_real * freq_i + x_imag * freq_r

    x = x.clone()  # Don't modify input for testing
    x[..., 0::2] = out_real.to(x.dtype)
    x[..., 1::2] = out_imag.to(x.dtype)
    return x


def test_rotary_emb_2d():
    """Test rotary embedding with 2D input (batch, dim)."""
    torch.manual_seed(42)
    dim = 64
    seqlen = 32
    batch_size = 8

    freqs_cis = precompute_freqs_cis(dim, seqlen, 0, 10000.0, 1.0, 32, 1)
    x = torch.randn(batch_size, dim)

    ref = apply_rotary_emb_reference(x, freqs_cis[:batch_size])
    native = apply_rotary_emb_pytorch_native(x, freqs_cis[:batch_size])

    max_err = (ref - native).abs().max().item()
    print(f"  rotary_emb 2D: max error = {max_err:.2e}")
    assert torch.allclose(ref, native, atol=1e-5), f"Max error: {max_err}"


def test_rotary_emb_3d():
    """Test rotary embedding with 3D input (batch, heads, dim)."""
    torch.manual_seed(42)
    dim = 64
    seqlen = 32
    batch_size = 4
    n_heads = 8

    freqs_cis = precompute_freqs_cis(dim, seqlen, 0, 10000.0, 1.0, 32, 1)
    x = torch.randn(batch_size, n_heads, dim)

    ref = apply_rotary_emb_reference(x, freqs_cis[:batch_size])
    native = apply_rotary_emb_pytorch_native(x, freqs_cis[:batch_size])

    max_err = (ref - native).abs().max().item()
    print(f"  rotary_emb 3D: max error = {max_err:.2e}")
    assert torch.allclose(ref, native, atol=1e-5), f"Max error: {max_err}"


def test_rotary_emb_with_positions():
    """Test rotary embedding with explicit position indices."""
    torch.manual_seed(42)
    dim = 64
    seqlen = 128
    batch_size = 4

    freqs_cis = precompute_freqs_cis(dim, seqlen, 0, 10000.0, 1.0, 32, 1)
    positions = torch.tensor([0, 5, 10, 127])
    x = torch.randn(batch_size, dim)

    ref = apply_rotary_emb_reference(x, freqs_cis, positions)
    native = apply_rotary_emb_pytorch_native(x, freqs_cis, positions)

    max_err = (ref - native).abs().max().item()
    print(f"  rotary_emb positions: max error = {max_err:.2e}")
    assert torch.allclose(ref, native, atol=1e-5), f"Max error: {max_err}"


def test_rotary_emb_inverse():
    """Test that applying forward then inverse gives back the original."""
    torch.manual_seed(42)
    dim = 64
    seqlen = 32
    batch_size = 4

    freqs_cis = precompute_freqs_cis(dim, seqlen, 0, 10000.0, 1.0, 32, 1)
    x_orig = torch.randn(batch_size, dim)

    x_fwd = apply_rotary_emb_pytorch_native(x_orig, freqs_cis[:batch_size], inverse=False)
    x_inv = apply_rotary_emb_pytorch_native(x_fwd, freqs_cis[:batch_size], inverse=True)

    max_err = (x_orig - x_inv).abs().max().item()
    print(f"  rotary_emb inverse roundtrip: max error = {max_err:.2e}")
    assert torch.allclose(x_orig, x_inv, atol=1e-5), f"Roundtrip error: {max_err}"


def test_rotary_emb_3d_with_positions():
    """Test rotary embedding 3D with positions."""
    torch.manual_seed(42)
    dim = 128
    seqlen = 256
    batch_size = 4
    n_heads = 16

    freqs_cis = precompute_freqs_cis(dim, seqlen, 0, 10000.0, 1.0, 32, 1)
    positions = torch.tensor([0, 100, 200, 255])
    x = torch.randn(batch_size, n_heads, dim)

    ref = apply_rotary_emb_reference(x, freqs_cis, positions)
    native = apply_rotary_emb_pytorch_native(x, freqs_cis, positions)

    max_err = (ref - native).abs().max().item()
    print(f"  rotary_emb 3D+positions: max error = {max_err:.2e}")
    assert torch.allclose(ref, native, atol=1e-5), f"Max error: {max_err}"


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    tests = [
        # act_quant
        ("act_quant_basic", test_act_quant_basic),
        ("act_quant_scale_range", test_act_quant_scale_range),
        ("act_quant_with_scale_fmt", test_act_quant_with_scale_fmt),
        ("act_quant_multidim", test_act_quant_multidim),
        # fused_scale
        ("fused_scale_basic", test_fused_scale_basic),
        ("fused_scale_dtypes", test_fused_scale_dtypes),
        # topk_transform
        ("topk_small_seq", test_topk_small_seq),
        ("topk_large_seq", test_topk_large_seq),
        ("topk_page_translation", test_topk_page_translation),
        # compressed_metadata
        ("compressed_metadata_c4", test_compressed_metadata_c4),
        ("compressed_metadata_c128", test_compressed_metadata_c128),
        ("compressed_metadata_page_indices", test_compressed_metadata_page_indices),
        # rotary_emb
        ("rotary_emb_2d", test_rotary_emb_2d),
        ("rotary_emb_3d", test_rotary_emb_3d),
        ("rotary_emb_with_positions", test_rotary_emb_with_positions),
        ("rotary_emb_inverse", test_rotary_emb_inverse),
        ("rotary_emb_3d_with_positions", test_rotary_emb_3d_with_positions),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"Running {name}...")
            test_fn()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        exit(1)
    else:
        print("All tests passed!")
