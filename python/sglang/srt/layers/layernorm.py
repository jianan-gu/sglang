# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.utils import is_cuda_available

_is_cuda = is_cuda_available()

if _is_cuda:
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )

from sglang.srt.cpu_utils import cpu_has_amx_support

if cpu_has_amx_support():
    import sgl_kernel.cpu
    import sgl_kernel

from sglang.srt.custom_op import CustomOp

logger = logging.getLogger(__name__)

class L2Norm(CustomOp):
    def __init__(
        self,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self._has_amx_support = cpu_has_amx_support()
        if self._has_amx_support:
            self.cpu_l2norm = sgl_kernel.common_ops.l2norm_cpu

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        return x

    def forward_cpu(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._has_amx_support:
            eps = self.variance_epsilon
            return self.cpu_l2norm(x, eps)
        else:
            return self.forward_native(x)

class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        self._has_amx_support = cpu_has_amx_support()
        if self._has_amx_support:
            self.cpu_fused_add_rmsnorm = sgl_kernel.cpu.fused_add_rmsnorm
            self.cpu_rmsnorm = sgl_kernel.cpu.rmsnorm

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._has_amx_support:
            weight_data = self.weight.data
            eps = self.variance_epsilon

            if residual is not None:
                self.cpu_fused_add_rmsnorm(x, residual, weight_data, eps)
                return x, residual
            return self.cpu_rmsnorm(x, weight_data, eps)
        else:
            return self.forward_native(x, residual)


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

if not (_is_cuda or (not _is_cuda and cpu_has_amx_support())):
    logger.info(
        "sgl-kernel is not available on Non-NV platforms or Non-AMX CPUs. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
