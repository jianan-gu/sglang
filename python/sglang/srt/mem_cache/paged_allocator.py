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
PyTorch native implementations of paged memory allocation kernels and allocator.

These are pure-Python/PyTorch equivalents of the Triton kernels alloc_extend_kernel
and alloc_decode_kernel, plus the PagedTokenToKVPoolAllocator class that uses them
for paged KV-cache memory management.
"""

import logging

import torch

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


def _ceil_div(a, b):
    """Integer ceiling division. Works for both Python ints and torch Tensors."""
    return (a + b - 1) // b


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

    For each request in the batch, fills out_indices with the physical token
    locations for newly extended tokens, using paged allocation.

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
        seq_lens: (bs,) sequence lengths *after* the decode step
                  (i.e. already incremented by 1).
        last_loc: (bs,) the last allocated physical location per request.
        free_pages: (num_free_pages,) free page indices.
        out_indices: (bs,) output tensor to fill with the new physical location.
        page_size: number of token slots per page.
    """
    pre_lens = seq_lens - 1

    # Compute number of new pages needed per request (0 or 1)
    num_pages_after = _ceil_div(seq_lens, page_size)
    num_pages_before = _ceil_div(pre_lens, page_size)
    num_new_pages = num_pages_after - num_pages_before

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


class PagedTokenToKVPoolAllocator:
    """
    An allocator managing the indices to kv cache data with paged allocation.

    This is a PyTorch-native replacement for the Triton-based version.
    The output of one request is always page-aligned.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.seen_max_num_extend_tokens_next_power_of_2 = 1
        self.clear()

    def available_size(self):
        return len(self.free_pages) * self.page_size

    def merge_and_sort_free(self):
        if self.release_pages.numel() > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.release_pages = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
        self.free_pages = torch.sort(self.free_pages).values

    def alloc(self, need_size: int):
        """Page-aligned allocation, returning contiguous indices of pages."""
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        self.seen_max_num_extend_tokens_next_power_of_2 = max(
            self.seen_max_num_extend_tokens_next_power_of_2,
            next_power_of_2(extend_num_tokens),
        )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_kernel_pytorch(
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel_pytorch(
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat(
                    (free_page_indices, self.release_pages)
                )
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
