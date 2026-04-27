"""
Test script for flash_mla_with_kvcache CPU implementation.

This verifies correctness by comparing against a pure PyTorch reference.
"""
import torch
import math


def flash_mla_with_kvcache_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float = 0.0,
    causal: bool = False,
) -> tuple:
    """
    Pure PyTorch reference implementation of flash_mla_with_kvcache.

    q:              [batch_size, seq_len_q, num_heads_q, head_dim]
    k_cache:        [num_blocks, page_block_size, num_heads_k, head_dim]
    block_table:    [batch_size, max_num_blocks_per_seq], int32
    cache_seqlens:  [batch_size], int32
    head_dim_v:     int (512 for DeepSeek V3)
    softmax_scale:  float
    causal:         bool

    Returns:
        out:          [batch_size, seq_len_q, num_heads_q, head_dim_v]
        softmax_lse:  [batch_size, num_heads_q, seq_len_q], float32
    """
    batch_size, seq_len_q, num_heads_q, head_dim = q.shape
    page_block_size = k_cache.shape[1]
    num_heads_k = k_cache.shape[2]
    num_groups = num_heads_q // num_heads_k

    if softmax_scale == 0.0:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    out = torch.zeros(batch_size, seq_len_q, num_heads_q, head_dim_v, dtype=q.dtype)
    softmax_lse = torch.full(
        (batch_size, num_heads_q, seq_len_q), -float("inf"), dtype=torch.float32
    )

    for b in range(batch_size):
        seq_len_kv = int(cache_seqlens[b].item())

        # Gather all KV tokens from paged cache
        kv_tokens = []
        for kv_pos in range(seq_len_kv):
            block_idx = kv_pos // page_block_size
            block_offset = kv_pos % page_block_size
            physical_block = int(block_table[b, block_idx].item())
            kv_tokens.append(k_cache[physical_block, block_offset])
        # kv_tokens: list of [num_heads_k, head_dim]

        if seq_len_kv == 0:
            continue

        # Stack to [seq_len_kv, num_heads_k, head_dim]
        kv = torch.stack(kv_tokens, dim=0)

        for sq in range(seq_len_q):
            num_keys = min(sq + 1, seq_len_kv) if causal else seq_len_kv

            for h in range(num_heads_q):
                h_kv = h // num_groups

                # q_vec: [head_dim]
                q_vec = q[b, sq, h].float()

                # k_mat: [num_keys, head_dim]
                k_mat = kv[:num_keys, h_kv].float()

                # v_mat: [num_keys, head_dim_v] - first head_dim_v elements
                v_mat = kv[:num_keys, h_kv, :head_dim_v].float()

                # Attention scores: [num_keys]
                scores = (q_vec @ k_mat.t()) * softmax_scale

                # Softmax
                max_score = scores.max()
                exp_scores = torch.exp(scores - max_score)
                sum_exp = exp_scores.sum()

                # Output
                attn_weights = exp_scores / sum_exp
                out_vec = attn_weights @ v_mat

                out[b, sq, h] = out_vec.to(q.dtype)
                softmax_lse[b, h, sq] = max_score + torch.log(sum_exp)

    return out, softmax_lse


def test_flash_mla_reference():
    """Test the reference implementation with small dimensions."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len_q = 1  # decode scenario
    num_heads_q = 4
    num_heads_k = 1
    head_dim = 64  # smaller for testing
    head_dim_v = 48
    page_block_size = 4
    max_seq_len = 16

    max_num_blocks = (max_seq_len + page_block_size - 1) // page_block_size
    num_blocks = batch_size * max_num_blocks + 4  # some extra blocks

    # Create inputs
    q = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, page_block_size, num_heads_k, head_dim, dtype=torch.bfloat16)
    block_table = torch.zeros(batch_size, max_num_blocks, dtype=torch.int32)
    cache_seqlens = torch.tensor([12, 8], dtype=torch.int32)

    # Assign physical blocks
    block_counter = 0
    for b in range(batch_size):
        num_needed = (int(cache_seqlens[b].item()) + page_block_size - 1) // page_block_size
        for i in range(num_needed):
            block_table[b, i] = block_counter
            block_counter += 1

    out, lse = flash_mla_with_kvcache_reference(
        q, k_cache, block_table, cache_seqlens, head_dim_v
    )

    print(f"Reference output shape: {out.shape}")
    print(f"Reference LSE shape: {lse.shape}")
    print(f"Reference output sample: {out[0, 0, 0, :5]}")
    print(f"Reference LSE sample: {lse[0, 0, 0]}")

    # Verify shapes
    assert out.shape == (batch_size, seq_len_q, num_heads_q, head_dim_v)
    assert lse.shape == (batch_size, num_heads_q, seq_len_q)
    print("Reference test passed!")


def test_flash_mla_cpu_kernel():
    """Test the CPU kernel against reference (requires compiled extension)."""
    try:
        import sgl_kernel

        has_kernel = hasattr(sgl_kernel.ops, "flash_mla_with_kvcache_cpu")
    except (ImportError, AttributeError):
        try:
            # Try loading via torch.ops
            _ = torch.ops.sgl_kernel.flash_mla_with_kvcache_cpu
            has_kernel = True
        except (RuntimeError, AttributeError):
            has_kernel = False

    if not has_kernel:
        print("CPU kernel not available, skipping kernel test.")
        print("To test, compile the extension first:")
        print("  cd sgl-kernel/csrc/cpu && mkdir build && cd build")
        print("  cmake .. && make -j")
        return

    torch.manual_seed(42)

    batch_size = 2
    seq_len_q = 1
    num_heads_q = 4
    num_heads_k = 1
    head_dim = 64
    head_dim_v = 48
    page_block_size = 4
    max_seq_len = 16
    max_num_blocks = (max_seq_len + page_block_size - 1) // page_block_size
    num_blocks = batch_size * max_num_blocks + 4

    q = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, page_block_size, num_heads_k, head_dim, dtype=torch.bfloat16)
    block_table = torch.zeros(batch_size, max_num_blocks, dtype=torch.int32)
    cache_seqlens = torch.tensor([12, 8], dtype=torch.int32)

    block_counter = 0
    for b in range(batch_size):
        num_needed = (int(cache_seqlens[b].item()) + page_block_size - 1) // page_block_size
        for i in range(num_needed):
            block_table[b, i] = block_counter
            block_counter += 1

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Reference
    ref_out, ref_lse = flash_mla_with_kvcache_reference(
        q, k_cache, block_table, cache_seqlens, head_dim_v, softmax_scale
    )

    # CPU kernel
    cpu_out, cpu_lse = torch.ops.sgl_kernel.flash_mla_with_kvcache_cpu(
        q, k_cache, block_table, cache_seqlens, head_dim_v, softmax_scale, False
    )

    # Compare
    out_diff = (ref_out.float() - cpu_out.float()).abs().max().item()
    lse_diff = (ref_lse - cpu_lse).abs().max().item()

    print(f"Output max diff: {out_diff}")
    print(f"LSE max diff: {lse_diff}")
    assert out_diff < 0.05, f"Output diff too large: {out_diff}"
    assert lse_diff < 0.1, f"LSE diff too large: {lse_diff}"
    print("CPU kernel test passed!")


def test_flash_mla_deepseek_dims():
    """Test with DeepSeek V3 dimensions."""
    torch.manual_seed(42)

    batch_size = 1
    seq_len_q = 1
    num_heads_q = 16  # smaller than actual 128 for testing
    num_heads_k = 1
    head_dim = 576
    head_dim_v = 512
    page_block_size = 64
    max_seq_len = 256

    max_num_blocks = (max_seq_len + page_block_size - 1) // page_block_size
    num_blocks = max_num_blocks + 2

    q = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, page_block_size, num_heads_k, head_dim, dtype=torch.bfloat16)
    block_table = torch.arange(max_num_blocks, dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([200], dtype=torch.int32)

    out, lse = flash_mla_with_kvcache_reference(
        q, k_cache, block_table, cache_seqlens, head_dim_v
    )

    print(f"DeepSeek dims - output shape: {out.shape}")
    print(f"DeepSeek dims - LSE shape: {lse.shape}")
    assert out.shape == (1, 1, num_heads_q, head_dim_v)
    assert lse.shape == (1, num_heads_q, 1)
    print("DeepSeek dimensions test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing flash_mla_with_kvcache reference implementation")
    print("=" * 60)
    test_flash_mla_reference()
    print()

    print("=" * 60)
    print("Testing with DeepSeek V3 dimensions")
    print("=" * 60)
    test_flash_mla_deepseek_dims()
    print()

    print("=" * 60)
    print("Testing CPU kernel (if compiled)")
    print("=" * 60)
    test_flash_mla_cpu_kernel()
    print()

    print("All tests passed!")
