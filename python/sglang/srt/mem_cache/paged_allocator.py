"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
PyTorch native implementations of paged memory allocation kernels.

These are pure-Python equivalents of the Triton kernels alloc_extend_kernel
and alloc_decode_kernel, used for paged KV-cache memory management.
"""

import torch


def _ceil_div(a, b):
    """Integer ceiling division."""
    return (a + b - 1) // b


def alloc_extend_kernel_pytorch(
    pre_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    page_size: int,
):
    """
    PyTorch native version of alloc_extend_kernel.

    For each request in the batch, this function fills out_indices with the
    physical token locations for newly extended tokens, using paged allocation.

    Each request's output is divided into up to three parts:
      Part 1: Tokens filling the remainder of the last partially-used page.
      Part 2: Tokens filling new full pages.
      Part 3: Tokens starting a new partial page at the end.

    Args:
        pre_lens: (bs,) prefix lengths (tokens already allocated).
        seq_lens: (bs,) total sequence lengths after extension.
        last_loc: (bs,) the last allocated physical location per request.
        free_pages: (num_free_pages,) free page indices.
        out_indices: (extend_num_tokens,) output tensor to fill with physical locations.
        page_size: number of token slots per page.
    """
    bs = pre_lens.shape[0]
    extend_lens = seq_lens - pre_lens

    # Compute prefix-sum of extend_lens to find each request's output start
    extend_cumsum = torch.cumsum(extend_lens, dim=0)
    output_start_locs = extend_cumsum - extend_lens  # exclusive prefix sum

    # Compute number of new pages needed per request
    num_pages_after = _ceil_div(seq_lens, page_size)
    num_pages_before = _ceil_div(pre_lens, page_size)
    num_new_pages = num_pages_after - num_pages_before

    # Compute prefix-sum of new pages to find each request's free_page offset
    new_pages_cumsum = torch.cumsum(num_new_pages, dim=0)
    new_page_start_locs = new_pages_cumsum - num_new_pages  # exclusive prefix sum

    for i in range(bs):
        pre_len = pre_lens[i].item()
        seq_len = seq_lens[i].item()
        extend_len = seq_len - pre_len
        if extend_len == 0:
            continue

        output_start = output_start_locs[i].item()
        new_page_start = new_page_start_locs[i].item()
        num_new_pages_self = num_new_pages[i].item()
        last_loc_i = last_loc[i].item()

        # Part 1: fill the old partial page
        # How many tokens fit in the remainder of the current last page
        page_boundary = _ceil_div(pre_len, page_size) * page_size
        num_part1 = min(seq_len, page_boundary) - pre_len
        if num_part1 > 0:
            offsets = torch.arange(num_part1, device=out_indices.device)
            out_indices[output_start : output_start + num_part1] = (
                last_loc_i + 1 + offsets
            )

        if pre_len + num_part1 == seq_len:
            continue

        # Part 2: fill new full pages
        full_page_end = (seq_len // page_size) * page_size
        full_page_start = _ceil_div(pre_len, page_size) * page_size
        num_part2 = full_page_end - full_page_start
        if num_part2 > 0:
            offsets = torch.arange(num_part2, device=out_indices.device)
            page_indices = offsets // page_size  # which new page each offset falls into
            in_page_offsets = offsets % page_size
            pages = free_pages[new_page_start + page_indices]
            out_indices[
                output_start + num_part1 : output_start + num_part1 + num_part2
            ] = (pages * page_size + in_page_offsets)

        if pre_len + num_part1 + num_part2 == seq_len:
            continue

        # Part 3: fill the new partial page at the end
        num_part3 = seq_len - (seq_len // page_size) * page_size
        if num_part3 > 0:
            start_page = free_pages[new_page_start + num_new_pages_self - 1].item()
            offsets = torch.arange(num_part3, device=out_indices.device)
            out_indices[
                output_start
                + num_part1
                + num_part2 : output_start
                + num_part1
                + num_part2
                + num_part3
            ] = (start_page * page_size + offsets)


def alloc_decode_kernel_pytorch(
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    page_size: int,
):
    """
    PyTorch native version of alloc_decode_kernel.

    For each request in the batch, allocates exactly one new token slot.
    If the new token fits in the current page, uses last_loc + 1.
    If a new page is needed, takes the next free page.

    Args:
        seq_lens: (bs,) sequence lengths *after* the decode step (i.e. already incremented by 1).
        last_loc: (bs,) the last allocated physical location per request.
        free_pages: (num_free_pages,) free page indices.
        out_indices: (bs,) output tensor to fill with the new physical location.
        page_size: number of token slots per page.
    """
    bs = seq_lens.shape[0]
    pre_lens = seq_lens - 1

    # Compute number of new pages needed per request
    num_pages_after = _ceil_div(seq_lens, page_size)
    num_pages_before = _ceil_div(pre_lens, page_size)
    num_new_pages = num_pages_after - num_pages_before  # 0 or 1 per request

    # Compute prefix-sum for free_page offsets
    new_pages_cumsum = torch.cumsum(num_new_pages, dim=0)
    new_page_start_locs = new_pages_cumsum - num_new_pages

    # Requests that don't need a new page: next slot is last_loc + 1
    no_new_page_mask = num_new_pages == 0
    out_indices[no_new_page_mask] = last_loc[no_new_page_mask] + 1

    # Requests that need a new page: start of the allocated free page
    new_page_mask = num_new_pages > 0
    if new_page_mask.any():
        new_page_offsets = new_page_start_locs[new_page_mask]
        pages = free_pages[new_page_offsets]
        out_indices[new_page_mask] = pages.to(out_indices.dtype) * page_size


def get_num_new_pages(
    seq_lens: torch.Tensor,
    page_size: int,
    prefix_lens: torch.Tensor = None,
    decode: bool = False,
) -> int:
    """
    Compute total number of new pages needed across all requests.

    Args:
        seq_lens: sequence lengths (CPU tensor).
        page_size: tokens per page.
        prefix_lens: prefix lengths (CPU tensor). Required when decode=False.
        decode: if True, each request extends by 1 token.

    Returns:
        Total number of new pages as an integer.
    """
    if decode:
        pre_lens = seq_lens - 1
    else:
        pre_lens = prefix_lens

    num_pages_after = _ceil_div(seq_lens, page_size)
    num_pages_before = _ceil_div(pre_lens, page_size)
    return (num_pages_after - num_pages_before).sum().item()


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1
