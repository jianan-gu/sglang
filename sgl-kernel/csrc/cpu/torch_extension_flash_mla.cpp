/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/all.h>
#include <torch/extension.h>
#include <torch/library.h>

// flash_mla_with_kvcache
std::tuple<at::Tensor, at::Tensor> flash_mla_with_kvcache_cpu(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    int64_t head_dim_v,
    double softmax_scale,
    bool causal);

// Register with torch.ops for use via TORCH_LIBRARY
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "flash_mla_with_kvcache_cpu(Tensor q, Tensor k_cache, Tensor block_table, "
      "Tensor cache_seqlens, int head_dim_v, float softmax_scale, bool causal) -> (Tensor, Tensor)");
  m.impl("flash_mla_with_kvcache_cpu", torch::kCPU, &flash_mla_with_kvcache_cpu);
}

// pybind11 module for JIT compilation / direct import
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_mla_with_kvcache_cpu", &flash_mla_with_kvcache_cpu,
        "Flash MLA with KV Cache - CPU optimized implementation",
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("block_table"),
        py::arg("cache_seqlens"),
        py::arg("head_dim_v"),
        py::arg("softmax_scale"),
        py::arg("causal"));
}
