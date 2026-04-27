/*****************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * CPU implementation of flash_mla_with_kvcache for DeepSeek MLA architecture.
 *
 * This implements the dense decode attention path with paged KV cache, following
 * the FlashMLA interface (https://github.com/deepseek-ai/FlashMLA).
 *
 * The implementation uses flash attention algorithm with online softmax and
 * is optimized for CPU with vectorized operations.
 *
 * For DeepSeek V3 MLA:
 *   - head_dim (key) = 576 = head_dim_v (512) + rope_dim (64)
 *   - num_heads_k = 1 (single KV head, KV is compressed latent)
 *   - K and V share the same cache tensor; V = k_cache[..., :head_dim_v]
 *
 ****************************************************************************************/
#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a_fvec0 = fVec::loadu(acc + d) * s_fvec;
    fVec a_fvec1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * s);
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void copy_stub<at::BFloat16>(
    at::BFloat16* __restrict__ out, const float* __restrict__ acc, float s, int64_t size) {
  const __m512 vscale = _mm512_set1_ps(s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - 32; d += 32) {
    __m512 va0 = _mm512_mul_ps(_mm512_loadu_ps(acc + d), vscale);
    __m512 va1 = _mm512_mul_ps(_mm512_loadu_ps(acc + d + 16), vscale);
    __m512i vb = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
    _mm512_storeu_si512(out + d, vb);
  }
  int remainder = size - d;
  if (remainder > 0) {
    if (remainder <= 16) {
      const __mmask16 vmask = (1ULL << remainder) - 1;
      __m512 va = _mm512_mul_ps(_mm512_maskz_loadu_ps(vmask, acc + d), vscale);
      __m256i vb = (__m256i)(_mm512_cvtneps_pbh(va));
      _mm256_mask_storeu_epi16(reinterpret_cast<__m256i*>(out + d), vmask, vb);
    } else {
      const __mmask16 vmask = (1ULL << (remainder - 16)) - 1;
      __m512 va0 = _mm512_mul_ps(_mm512_loadu_ps(acc + d), vscale);
      __m512 va1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(vmask, acc + d + 16), vscale);
      __m512i vb = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
      const __mmask32 vmask2 = (1ULL << remainder) - 1;
      _mm512_mask_storeu_epi16(reinterpret_cast<__m512i*>(out + d), vmask2, vb);
    }
  }
}
#endif

// Compute dot product: q[head_dim] . k[head_dim] * scale
// Returns a single float: the attention score
template <typename scalar_t>
inline float dot_product_scaled(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ k_ptr,
    int64_t head_dim,
    float scale) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();

  fVec acc0 = fVec(0.f);
  fVec acc1 = fVec(0.f);
  int64_t d = 0;
  for (; d <= head_dim - kVecSize; d += kVecSize) {
    bVec q_bvec = bVec::loadu(q_ptr + d);
    bVec k_bvec = bVec::loadu(k_ptr + d);
    fVec q0, q1, k0, k1;
    std::tie(q0, q1) = at::vec::convert_to_float(q_bvec);
    std::tie(k0, k1) = at::vec::convert_to_float(k_bvec);
    acc0 = acc0 + q0 * k0;
    acc1 = acc1 + q1 * k1;
  }
  float sum = vec_reduce_sum(acc0) + vec_reduce_sum(acc1);
  for (; d < head_dim; ++d) {
    sum += static_cast<float>(q_ptr[d]) * static_cast<float>(k_ptr[d]);
  }
  return sum * scale;
}

#if defined(CPU_CAPABILITY_AVX512)
// Specialized dot product for BFloat16 using AMX-style dpbf16
inline float dot_product_scaled_bf16(
    const at::BFloat16* __restrict__ q_ptr,
    const at::BFloat16* __restrict__ k_ptr,
    int64_t head_dim,
    float scale) {
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  int64_t d = 0;
  for (; d <= head_dim - 32; d += 32) {
    __m512bh vq = (__m512bh)_mm512_loadu_si512(q_ptr + d);
    __m512bh vk = (__m512bh)_mm512_loadu_si512(k_ptr + d);
    vacc0 = _mm512_dpbf16_ps(vacc0, vq, vk);
  }
  int64_t count = head_dim - d;
  if (count > 0) {
    __mmask32 mask = (1ULL << count) - 1;
    __m512bh vq = (__m512bh)_mm512_maskz_loadu_epi16(mask, q_ptr + d);
    __m512bh vk = (__m512bh)_mm512_maskz_loadu_epi16(mask, k_ptr + d);
    vacc1 = _mm512_dpbf16_ps(vacc1, vq, vk);
  }
  return (_mm512_reduce_add_ps(vacc0) + _mm512_reduce_add_ps(vacc1)) * scale;
}
#endif

// Compute weighted accumulation: v_prime[head_dim_v] += weight * v_ptr[head_dim_v]
template <typename scalar_t>
inline void weighted_accumulate(
    float* __restrict__ v_prime,
    const scalar_t* __restrict__ v_ptr,
    float weight,
    int64_t head_dim_v) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();
  const fVec w_vec = fVec(weight);

  int64_t d = 0;
  for (; d <= head_dim_v - kVecSize; d += kVecSize) {
    bVec v_bvec = bVec::loadu(v_ptr + d);
    fVec v0, v1;
    std::tie(v0, v1) = at::vec::convert_to_float(v_bvec);
    fVec a0 = fVec::loadu(v_prime + d);
    fVec a1 = fVec::loadu(v_prime + d + fVec::size());
    a0 = a0 + v0 * w_vec;
    a1 = a1 + v1 * w_vec;
    a0.store(v_prime + d);
    a1.store(v_prime + d + fVec::size());
  }
  for (; d < head_dim_v; ++d) {
    v_prime[d] += weight * static_cast<float>(v_ptr[d]);
  }
}

// Scale v_prime in-place
inline void scale_v_prime(float* __restrict__ v_prime, float scale, int64_t size) {
  using fVec = at::vec::Vectorized<float>;
  const fVec s_vec = fVec(scale);
  int64_t d = 0;
  for (; d <= size - fVec::size(); d += fVec::size()) {
    fVec v = fVec::loadu(v_prime + d) * s_vec;
    v.store(v_prime + d);
  }
  for (; d < size; ++d) {
    v_prime[d] *= scale;
  }
}

// ============================================================================
// Flash MLA with KV Cache - Dense Decode Attention
// ============================================================================
//
// This implements the core MLA decode attention with paged KV cache.
//
// Algorithm (per batch, per query position, per head):
//   1. For each KV token (gathered via block_table from paged cache):
//      a. Compute attention score: s = q @ k^T * scale  (using full head_dim)
//      b. Online softmax update:
//         - m_new = max(m_old, s)
//         - correction = exp(m_old - m_new)
//         - v_prime = v_prime * correction + exp(s - m_new) * v
//         - s_prime = s_prime * correction + exp(s - m_new)
//      c. Value is read from first head_dim_v elements of k_cache
//   2. Final output = v_prime / s_prime
//
// For MLA: K and V share the same cache.
//   k_cache[block, token, kv_head, :] has head_dim elements.
//   The first head_dim_v elements are the value (compressed latent).
//   All head_dim elements are used for the key.
//
template <typename scalar_t>
void flash_mla_decode_kernel_impl(
    scalar_t* __restrict__ output,       // [batch_size, seq_len_q, num_heads_q, head_dim_v]
    float* __restrict__ softmax_lse,     // [batch_size, num_heads_q, seq_len_q]
    const scalar_t* __restrict__ q,      // [batch_size, seq_len_q, num_heads_q, head_dim]
    const scalar_t* __restrict__ k_cache, // [num_blocks, page_block_size, num_heads_k, head_dim]
    const int32_t* __restrict__ block_table, // [batch_size, max_num_blocks_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [batch_size]
    int64_t batch_size,
    int64_t seq_len_q,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int64_t head_dim_v,
    int64_t page_block_size,
    int64_t max_num_blocks_per_seq,
    float sm_scale,
    bool causal,
    int64_t num_kv_splits,
    float* __restrict__ split_buf) {     // [batch_size, seq_len_q, num_heads_q, num_kv_splits, (head_dim_v + 1)]
  // Strides for q
  const int64_t q_stride_b = seq_len_q * num_heads_q * head_dim;
  const int64_t q_stride_s = num_heads_q * head_dim;
  const int64_t q_stride_h = head_dim;

  // Strides for k_cache
  const int64_t kc_stride_block = page_block_size * num_heads_k * head_dim;
  const int64_t kc_stride_token = num_heads_k * head_dim;
  const int64_t kc_stride_h = head_dim;

  // Strides for output
  const int64_t o_stride_b = seq_len_q * num_heads_q * head_dim_v;
  const int64_t o_stride_s = num_heads_q * head_dim_v;
  const int64_t o_stride_h = head_dim_v;

  // Strides for softmax_lse: [batch_size, num_heads_q, seq_len_q]
  const int64_t lse_stride_b = num_heads_q * seq_len_q;
  const int64_t lse_stride_h = seq_len_q;

  // Strides for split buffer: [batch_size, seq_len_q, num_heads_q, num_kv_splits, (head_dim_v + 1)]
  const int64_t sp_stride_b = seq_len_q * num_heads_q * num_kv_splits * (head_dim_v + 1);
  const int64_t sp_stride_s = num_heads_q * num_kv_splits * (head_dim_v + 1);
  const int64_t sp_stride_h = num_kv_splits * (head_dim_v + 1);
  const int64_t sp_stride_k = head_dim_v + 1;

  const int64_t num_groups = num_heads_q / num_heads_k;

  // Use KV-splits for parallelism when sequence length is long
  const bool use_kv_splits = (num_kv_splits > 1);

  // Total work items: [batch_size, seq_len_q, num_heads_q, num_kv_splits]
  const int64_t total_work = batch_size * seq_len_q * num_heads_q * num_kv_splits;

  at::parallel_for(0, total_work, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t b = idx;
      int64_t kv_split_id = b % num_kv_splits;
      b /= num_kv_splits;
      int64_t h = b % num_heads_q;
      b /= num_heads_q;
      int64_t sq = b % seq_len_q;
      b /= seq_len_q;

      int64_t h_kv = h / num_groups;
      int64_t seq_len_kv = cache_seqlens[b];

      // Get the query pointer for this (batch, seq_q, head)
      const scalar_t* __restrict__ q_ptr = q + b * q_stride_b + sq * q_stride_s + h * q_stride_h;

      // For causal masking (only applicable when seq_len_q > 1)
      int64_t num_keys = causal ? std::min(sq + 1, seq_len_kv) : seq_len_kv;

      // KV split range
      const int64_t kv_split_size = div_up(num_keys, num_kv_splits);
      const int64_t kv_start = kv_split_id * kv_split_size;
      const int64_t kv_end = std::min(kv_start + kv_split_size, num_keys);

      // Where to store results: either directly to output or to split buffer
      float* __restrict__ v_prime;
      if (use_kv_splits) {
        v_prime = split_buf + b * sp_stride_b + sq * sp_stride_s + h * sp_stride_h + kv_split_id * sp_stride_k;
      } else {
        // Allocate temporary buffer for v_prime on stack
        // For non-split case, we store to a thread-local buffer first
        v_prime = split_buf + b * sp_stride_b + sq * sp_stride_s + h * sp_stride_h;
      }

      fill_stub(v_prime, 0.f, head_dim_v);

      float m_prime = -std::numeric_limits<float>::infinity();
      float s_prime = 0.f;

      // Loop over KV tokens in the assigned split
      for (int64_t kv_pos = kv_start; kv_pos < kv_end; ++kv_pos) {
        // Compute the physical location in paged KV cache
        int64_t block_idx = kv_pos / page_block_size;
        int64_t block_offset = kv_pos % page_block_size;
        int32_t physical_block = block_table[b * max_num_blocks_per_seq + block_idx];

        const scalar_t* __restrict__ kv_ptr =
            k_cache + physical_block * kc_stride_block + block_offset * kc_stride_token + h_kv * kc_stride_h;

        // Compute attention score: s = q @ k^T * scale
        float s_i;
#if defined(CPU_CAPABILITY_AVX512)
        if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          s_i = dot_product_scaled_bf16(q_ptr, kv_ptr, head_dim, sm_scale);
        } else {
          s_i = dot_product_scaled(q_ptr, kv_ptr, head_dim, sm_scale);
        }
#else
        s_i = dot_product_scaled(q_ptr, kv_ptr, head_dim, sm_scale);
#endif

        // Online softmax update
        float m_new = std::max(m_prime, s_i);
        float correction = std::exp(m_prime - m_new);
        float p_i = std::exp(s_i - m_new);

        // v_prime = v_prime * correction + p_i * v
        scale_v_prime(v_prime, correction, head_dim_v);

        // Accumulate value: v is first head_dim_v elements of kv_ptr
        weighted_accumulate(v_prime, kv_ptr, p_i, head_dim_v);

        // Update running sum
        s_prime = s_prime * correction + p_i;
        m_prime = m_new;
      }

      if (use_kv_splits) {
        // Store log-sum-exp for later accumulation across splits
        if (kv_end > kv_start) {
          // Normalize v_prime by s_prime for this split
          float inv_s = 1.0f / s_prime;
          scale_v_prime(v_prime, inv_s, head_dim_v);
          v_prime[head_dim_v] = m_prime + std::log(s_prime);
        } else {
          v_prime[head_dim_v] = -std::numeric_limits<float>::infinity();
        }
      } else {
        // Single split: directly write output
        float inv_s = (s_prime > 0.f) ? (1.0f / s_prime) : 0.f;
        scalar_t* __restrict__ out_ptr = output + b * o_stride_b + sq * o_stride_s + h * o_stride_h;
        copy_stub<scalar_t>(out_ptr, v_prime, inv_s, head_dim_v);

        // Write softmax LSE
        softmax_lse[b * lse_stride_b + h * lse_stride_h + sq] =
            (s_prime > 0.f) ? (m_prime + std::log(s_prime)) : -std::numeric_limits<float>::infinity();
      }
    }
  });

  // Accumulate across KV splits if needed
  if (use_kv_splits) {
    const int64_t total_output = batch_size * seq_len_q * num_heads_q;
    at::parallel_for(0, total_output, 0, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        int64_t b = idx;
        int64_t h = b % num_heads_q;
        b /= num_heads_q;
        int64_t sq = b % seq_len_q;
        b /= seq_len_q;

        float* __restrict__ acc =
            split_buf + b * sp_stride_b + sq * sp_stride_s + h * sp_stride_h;

        float global_m = -std::numeric_limits<float>::infinity();
        float global_s = 0.f;

        // First pass: find global max
        for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
          float lse_k = (acc + kv_id * sp_stride_k)[head_dim_v];
          global_m = std::max(global_m, lse_k);
        }

        // Second pass: accumulate with correction
        // Use the first split's buffer as accumulator
        using fVec = at::vec::Vectorized<float>;
        float* __restrict__ result = acc;  // reuse first split

        // Initialize result to zero
        fill_stub(result, 0.f, head_dim_v);

        for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
          float* __restrict__ split_v = acc + kv_id * sp_stride_k;
          float lse_k = split_v[head_dim_v];

          if (lse_k > -std::numeric_limits<float>::infinity()) {
            float weight = std::exp(lse_k - global_m);
            global_s += weight;

            // result += weight * split_v
            int64_t d = 0;
            for (; d <= head_dim_v - fVec::size(); d += fVec::size()) {
              fVec r = fVec::loadu(result + d);
              fVec v = fVec::loadu(split_v + d);
              r = r + v * fVec(weight);
              r.store(result + d);
            }
            for (; d < head_dim_v; ++d) {
              result[d] += weight * split_v[d];
            }
          }
        }

        // Write final output
        float inv_s = (global_s > 0.f) ? (1.0f / global_s) : 0.f;
        scalar_t* __restrict__ out_ptr = output + b * o_stride_b + sq * o_stride_s + h * o_stride_h;
        copy_stub<scalar_t>(out_ptr, result, inv_s, head_dim_v);

        // Write softmax LSE
        softmax_lse[b * lse_stride_b + h * lse_stride_h + sq] =
            (global_s > 0.f) ? (global_m + std::log(global_s)) : -std::numeric_limits<float>::infinity();
      }
    });
  }
}

}  // anonymous namespace

// ============================================================================
// Public interface: flash_mla_with_kvcache
// ============================================================================
//
// q:              [batch_size, seq_len_q, num_heads_q, head_dim]
// k_cache:        [num_blocks, page_block_size, num_heads_k, head_dim]
// block_table:    [batch_size, max_num_blocks_per_seq], int32
// cache_seqlens:  [batch_size], int32
// head_dim_v:     int (512 for DeepSeek V3)
//
// Returns:
//   out:          [batch_size, seq_len_q, num_heads_q, head_dim_v]
//   softmax_lse:  [batch_size, num_heads_q, seq_len_q], float32
//
std::tuple<at::Tensor, at::Tensor> flash_mla_with_kvcache_cpu(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    int64_t head_dim_v,
    double softmax_scale,
    bool causal) {
  RECORD_FUNCTION(
      "sgl_kernel::flash_mla_with_kvcache_cpu",
      std::vector<c10::IValue>({q, k_cache, block_table, cache_seqlens, head_dim_v, softmax_scale, causal}));

  // Input validation
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_cache);
  CHECK_INPUT(block_table);
  CHECK_INPUT(cache_seqlens);
  CHECK_DIM(4, q);
  CHECK_DIM(4, k_cache);
  CHECK_DIM(2, block_table);
  CHECK_DIM(1, cache_seqlens);
  CHECK_EQ(block_table.scalar_type(), at::kInt);
  CHECK_EQ(cache_seqlens.scalar_type(), at::kInt);

  // Extract dimensions
  int64_t batch_size = q.size(0);
  int64_t seq_len_q = q.size(1);
  int64_t num_heads_q = q.size(2);
  int64_t head_dim = q.size(3);

  int64_t page_block_size = k_cache.size(1);
  int64_t num_heads_k = k_cache.size(2);
  int64_t head_dim_k = k_cache.size(3);

  int64_t max_num_blocks_per_seq = block_table.size(1);

  // Validate dimensions
  CHECK_EQ(head_dim, head_dim_k);
  CHECK_EQ(block_table.size(0), batch_size);
  CHECK_EQ(cache_seqlens.size(0), batch_size);
  TORCH_CHECK(head_dim_v > 0 && head_dim_v <= head_dim,
              "head_dim_v must be positive and <= head_dim. Got head_dim_v=", head_dim_v,
              ", head_dim=", head_dim);
  TORCH_CHECK(num_heads_q % num_heads_k == 0,
              "num_heads_q must be divisible by num_heads_k. Got num_heads_q=", num_heads_q,
              ", num_heads_k=", num_heads_k);

  // Compute softmax scale
  float sm_scale = static_cast<float>(softmax_scale);
  if (sm_scale == 0.f) {
    sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  }

  // Determine number of KV splits for parallelism
  // Find the max sequence length to determine split strategy
  int64_t max_seq_len = 0;
  const int32_t* cache_seqlens_ptr = cache_seqlens.data_ptr<int32_t>();
  for (int64_t b = 0; b < batch_size; ++b) {
    max_seq_len = std::max(max_seq_len, static_cast<int64_t>(cache_seqlens_ptr[b]));
  }

  int64_t num_threads = at::get_num_threads();
  int64_t base_work = batch_size * seq_len_q * num_heads_q;

  // Use KV splits when there's not enough parallelism from batches*heads
  int64_t num_kv_splits = 1;
  if (base_work < num_threads && max_seq_len > 256) {
    num_kv_splits = std::min(
        div_up(num_threads, std::max(base_work, int64_t(1))),
        div_up(max_seq_len, int64_t(128)));
    num_kv_splits = std::max(num_kv_splits, int64_t(1));
  }

  // Allocate output tensors
  at::Tensor out = at::empty({batch_size, seq_len_q, num_heads_q, head_dim_v}, q.options());
  at::Tensor softmax_lse = at::empty({batch_size, num_heads_q, seq_len_q}, q.options().dtype(at::kFloat));

  // Allocate split buffer
  at::Tensor split_buf = at::empty(
      {batch_size * seq_len_q * num_heads_q * num_kv_splits * (head_dim_v + 1)},
      q.options().dtype(at::kFloat));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "flash_mla_with_kvcache_cpu", [&] {
    flash_mla_decode_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        softmax_lse.data_ptr<float>(),
        q.data_ptr<scalar_t>(),
        k_cache.data_ptr<scalar_t>(),
        block_table.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        batch_size,
        seq_len_q,
        num_heads_q,
        num_heads_k,
        head_dim,
        head_dim_v,
        page_block_size,
        max_num_blocks_per_seq,
        sm_scale,
        causal,
        num_kv_splits,
        split_buf.data_ptr<float>());
  });

  return std::make_tuple(out, softmax_lse);
}
