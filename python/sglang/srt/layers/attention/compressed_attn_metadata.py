"""PyTorch native implementation of compressed attention metadata initialization.

This replaces the Triton kernel `_init_compressed_attn_metadata_kernel` with
pure PyTorch operations for better portability and readability.
"""

from typing import Optional, Tuple

import torch


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _init_compressed_attn_metadata_pytorch(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    bs = seq_lens.shape[0]
    device = seq_lens.device

    # --- c4 (compress-by-4) metadata ---
    c4_should_compress = (seq_lens % 4) == 0
    c4_out_loc = torch.where(
        c4_should_compress, raw_out_loc // 4, torch.zeros_like(raw_out_loc)
    )
    c4_positions = (positions & (~3)).to(torch.int32)
    c4_seq_lens_raw = (seq_lens // 4).to(torch.int32)
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1)

    # --- c128 (compress-by-128) metadata ---
    c128_should_compress = (seq_lens % 128) == 0
    c128_out_loc = torch.where(
        c128_should_compress, raw_out_loc // 128, torch.zeros_like(raw_out_loc)
    )
    c128_positions = (positions & (~127)).to(torch.int32)
    c128_seq_lens_raw = (seq_lens // 128).to(torch.int32)
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1)

    # --- page indices for c128 ---
    c128_page_indices: Optional[torch.Tensor] = None
    if compute_page_indices:
        assert (
            page_table is not None
        ), "page_table required when compute_page_indices=True"
        assert page_size > 0, "page_size required when compute_page_indices=True"

        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages

        # offsets: [c128_max_seq_len]
        offsets = torch.arange(c128_max_seq_len, device=device, dtype=torch.int32)

        # page_idx and offset_in_page for every position
        page_idx = offsets // c128_page_size  # [c128_max_seq_len]
        offset_in_page = offsets % c128_page_size  # [c128_max_seq_len]

        # Clamp page_idx for safe gather (values beyond max_pages will be masked later)
        page_idx_clamped = torch.clamp(page_idx, max=max_pages - 1)

        # Gather page table values for all batches: page_table is [bs, max_pages]
        # page_idx_clamped is [c128_max_seq_len], expand to [bs, c128_max_seq_len]
        page_idx_expanded = page_idx_clamped.unsqueeze(0).expand(bs, -1)
        page_table_vals = torch.gather(
            page_table, dim=1, index=page_idx_expanded.to(torch.int64)
        ).to(
            torch.int32
        )  # [bs, c128_max_seq_len]

        # c_page_indices_vals = page_table_vals * c128_page_size + offset_in_page
        c_page_indices_vals = page_table_vals * c128_page_size + offset_in_page.unsqueeze(0)

        # Mask: set to -1 where offsets >= c128_seq_lens_raw per batch
        # c128_seq_lens_raw: [bs], offsets: [c128_max_seq_len]
        valid_mask = offsets.unsqueeze(0) < c128_seq_lens_raw.unsqueeze(1)  # [bs, c128_max_seq_len]
        c128_page_indices = torch.where(
            valid_mask, c_page_indices_vals, torch.tensor(-1, dtype=torch.int32, device=device)
        )

    return (
        c4_out_loc.to(torch.int32),
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc.to(torch.int32),
        c128_positions,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


def init_compressed_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    return _init_compressed_attn_metadata_pytorch(
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        page_size,
        compute_page_indices,
    )
