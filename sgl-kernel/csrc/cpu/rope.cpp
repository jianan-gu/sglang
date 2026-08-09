#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
void rotary_embedding_3D_kernel_impl(
    scalar_t* __restrict__ query_out,
    scalar_t* __restrict__ key_out,
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t query_out_stride_s,
    int64_t key_out_stride_s,
    int64_t key_stride_s,
    int64_t query_stride_h,
    int64_t query_out_stride_h) {
  int64_t HR = rotary_dim;
  int64_t HK = rotary_dim;
  int64_t COFF = HR / 2;
  at::parallel_for(0, num_tokens * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t seq{0}, head_id{0};
    data_index_init(begin, seq, num_tokens, head_id, num_heads);
    for (int64_t i = begin; i < end; ++i) {
      int64_t in_offset_q = seq * query_stride_s + head_id * query_stride_h;
      int64_t out_offset_q = seq * query_out_stride_s + head_id * query_out_stride_h;
      int64_t out_offset_k = seq * key_out_stride_s;
      int64_t p = 0;
      scalar_t* sin_start = nullptr;
      scalar_t* cos_start = nullptr;
      // step 0) get the rotary position embedding for the current position
      p = positions[seq];
      sin_start = cos_sin_cache + p * HR + COFF;
      cos_start = cos_sin_cache + p * HR;
      // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
      // head of query/key
      for (int64_t h = 0; h < rotary_dim; h += 2) {
        scalar_t cos = cos_start[h >> 1];
        scalar_t sin = sin_start[h >> 1];
        scalar_t in1 = query[in_offset_q + h];
        scalar_t in2 = query[in_offset_q + h + 1];
        scalar_t out1 = in1 * cos - in2 * sin;
        scalar_t out2 = in2 * cos + in1 * sin;
        query_out[out_offset_q + h] = out1;
        query_out[out_offset_q + h + 1] = out2;
      }
      for (int64_t h = 0; h < HK; h += 2) {
        scalar_t cos = cos_start[h >> 1];
        scalar_t sin = sin_start[h >> 1];
        int64_t k_pe_offset = seq * key_stride_s;
        scalar_t in1_k = key[k_pe_offset + h];
        scalar_t in2_k = key[k_pe_offset + h + 1];
        scalar_t out1_k = in1_k * cos - in2_k * sin;
        scalar_t out2_k = in2_k * cos + in1_k * sin;
        key_out[out_offset_k + h] = out1_k;
        key_out[out_offset_k + h + 1] = out2_k;
      }
      // move to the next index
      data_index_step(seq, num_tokens, head_id, num_heads);
    }
  });
}

template <typename scalar_t>
void rotary_embedding_neox_4D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();

  int64_t embed_dim = rotary_dim / 2;
  bool flag = (embed_dim % bVecSize == 0);
  int64_t loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  auto compute_loop = [&](int64_t token_head, scalar_t* cache_ptr, scalar_t* qk) {
    int64_t j = 0;
    for (; j < loop_upper; j += bVecSize) {
      int64_t rot_offset = j;
      int64_t x_index = rot_offset;
      int64_t y_index = embed_dim + rot_offset;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      bVec _cos = bVec::loadu(cache_ptr + x_index);
      bVec _sin = bVec::loadu(cache_ptr + y_index);

      bVec _q_x = bVec::loadu(qk + out_x);
      bVec _q_y = bVec::loadu(qk + out_y);
      fVec _cos_0, _cos_1;
      std::tie(_cos_0, _cos_1) = at::vec::convert_to_float(_cos);
      fVec _sin_0, _sin_1;
      std::tie(_sin_0, _sin_1) = at::vec::convert_to_float(_sin);
      fVec _q_x_0, _q_x_1;
      std::tie(_q_x_0, _q_x_1) = at::vec::convert_to_float(_q_x);
      fVec _q_y_0, _q_y_1;
      std::tie(_q_y_0, _q_y_1) = at::vec::convert_to_float(_q_y);

      auto out1_0 = _q_x_0 * _cos_0 - _q_y_0 * _sin_0;
      auto out1_1 = _q_x_1 * _cos_1 - _q_y_1 * _sin_1;
      auto out1 = convert_from_float_ext<scalar_t>(out1_0, out1_1);
      out1.store(qk + out_x);

      auto out2_0 = _q_y_0 * _cos_0 + _q_x_0 * _sin_0;
      auto out2_1 = _q_y_1 * _cos_1 + _q_x_1 * _sin_1;
      auto out2 = convert_from_float_ext<scalar_t>(out2_0, out2_1);
      out2.store(qk + out_y);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        int64_t x_index = j;
        int64_t y_index = embed_dim + j;

        int64_t out_x = token_head + x_index;
        int64_t out_y = token_head + y_index;

        float _cos = cache_ptr[x_index];
        float _sin = cache_ptr[y_index];

        float _q_x = qk[out_x];
        float _q_y = qk[out_y];

        qk[out_x] = _q_x * _cos - _q_y * _sin;
        qk[out_y] = _q_y * _cos + _q_x * _sin;
      }
    }
  };

#pragma omp parallel for collapse(2)
  for (int64_t bs = 0; bs < batch_size; ++bs) {
    for (int64_t seq = 0; seq < seq_len; ++seq) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = bs * query_stride_b + seq * query_stride_s + head_idx * query_stride_h;
        compute_loop(token_head, cache_ptr, query);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = bs * key_stride_b + seq * key_stride_s + head_idx * key_stride_h;
        compute_loop(token_head, cache_ptr, key);
      }
    }
  }
}

// Generic neox-style rope kernel matching apply_rotary_pos_emb_native_eager:
// out-of-place, computed in float, cos/sin laid out as [outer_size, inner_size, head_size]
// and shared across the num_heads dimension of q/k.
template <typename scalar_t, typename cos_t>
void apply_rotary_pos_emb_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    const cos_t* __restrict__ cos,
    const cos_t* __restrict__ sin,
    int64_t outer_size,
    int64_t num_heads,
    int64_t inner_size,
    int64_t head_size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using cVec = at::vec::Vectorized<cos_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t bVecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();

  int64_t embed_dim = head_size / 2;
  bool flag = (embed_dim % bVecSize == 0);
  int64_t loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  auto compute_loop = [&](int64_t token_head, const cos_t* cos_ptr, const cos_t* sin_ptr) {
    int64_t j = 0;
    for (; j < loop_upper; j += bVecSize) {
      int64_t rot_offset = j;
      int64_t x_index = rot_offset;
      int64_t y_index = embed_dim + rot_offset;

      int64_t out_x = token_head + x_index;
      int64_t out_y = token_head + y_index;

      if constexpr (std::is_same_v<scalar_t, float>) {
        // scalar_t == cos_t == float: one fVec per iteration
        fVec _cos_x = fVec::loadu(cos_ptr + x_index);
        fVec _sin_x = fVec::loadu(sin_ptr + x_index);
        fVec _cos_y = fVec::loadu(cos_ptr + y_index);
        fVec _sin_y = fVec::loadu(sin_ptr + y_index);
        fVec _q_x = fVec::loadu(in + out_x);
        fVec _q_y = fVec::loadu(in + out_y);
        (_q_x * _cos_x - _q_y * _sin_x).store(out + out_x);
        (_q_y * _cos_y + _q_x * _sin_y).store(out + out_y);
      } else {
        fVec _cos_x_0, _cos_x_1, _sin_x_0, _sin_x_1, _cos_y_0, _cos_y_1, _sin_y_0, _sin_y_1;
        if constexpr (std::is_same_v<cos_t, float>) {
          _cos_x_0 = fVec::loadu(cos_ptr + x_index);
          _cos_x_1 = fVec::loadu(cos_ptr + x_index + fVecSize);
          _sin_x_0 = fVec::loadu(sin_ptr + x_index);
          _sin_x_1 = fVec::loadu(sin_ptr + x_index + fVecSize);
          _cos_y_0 = fVec::loadu(cos_ptr + y_index);
          _cos_y_1 = fVec::loadu(cos_ptr + y_index + fVecSize);
          _sin_y_0 = fVec::loadu(sin_ptr + y_index);
          _sin_y_1 = fVec::loadu(sin_ptr + y_index + fVecSize);
        } else {
          std::tie(_cos_x_0, _cos_x_1) = at::vec::convert_to_float(cVec::loadu(cos_ptr + x_index));
          std::tie(_sin_x_0, _sin_x_1) = at::vec::convert_to_float(cVec::loadu(sin_ptr + x_index));
          std::tie(_cos_y_0, _cos_y_1) = at::vec::convert_to_float(cVec::loadu(cos_ptr + y_index));
          std::tie(_sin_y_0, _sin_y_1) = at::vec::convert_to_float(cVec::loadu(sin_ptr + y_index));
        }

        fVec _q_x_0, _q_x_1, _q_y_0, _q_y_1;
        std::tie(_q_x_0, _q_x_1) = at::vec::convert_to_float(bVec::loadu(in + out_x));
        std::tie(_q_y_0, _q_y_1) = at::vec::convert_to_float(bVec::loadu(in + out_y));

        auto out1_0 = _q_x_0 * _cos_x_0 - _q_y_0 * _sin_x_0;
        auto out1_1 = _q_x_1 * _cos_x_1 - _q_y_1 * _sin_x_1;
        auto out2_0 = _q_y_0 * _cos_y_0 + _q_x_0 * _sin_y_0;
        auto out2_1 = _q_y_1 * _cos_y_1 + _q_x_1 * _sin_y_1;
        convert_from_float_ext<scalar_t>(out1_0, out1_1).store(out + out_x);
        convert_from_float_ext<scalar_t>(out2_0, out2_1).store(out + out_y);
      }
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        int64_t x_index = j;
        int64_t y_index = embed_dim + j;

        int64_t out_x = token_head + x_index;
        int64_t out_y = token_head + y_index;

        float _cos_x = cos_ptr[x_index];
        float _sin_x = sin_ptr[x_index];
        float _cos_y = cos_ptr[y_index];
        float _sin_y = sin_ptr[y_index];

        float _q_x = in[out_x];
        float _q_y = in[out_y];

        out[out_x] = _q_x * _cos_x - _q_y * _sin_x;
        out[out_y] = _q_y * _cos_y + _q_x * _sin_y;
      }
    }
  };

  int64_t grain_size = std::max<int64_t>(GRAIN_SIZE / head_size, 1);
  at::parallel_for(0, outer_size * num_heads * inner_size, grain_size, [&](int64_t begin, int64_t end) {
    int64_t b{0}, h{0}, s{0};
    data_index_init(begin, b, outer_size, h, num_heads, s, inner_size);
    for (int64_t i = begin; i < end; ++i) {
      const cos_t* cos_ptr = cos + (b * inner_size + s) * head_size;
      const cos_t* sin_ptr = sin + (b * inner_size + s) * head_size;
      int64_t token_head = ((b * num_heads + h) * inner_size + s) * head_size;
      compute_loop(token_head, cos_ptr, sin_ptr);
      data_index_step(b, outer_size, h, num_heads, s, inner_size);
    }
  });
}

template <typename scalar_t>
inline scalar_t* get_cache_ptr(
    int64_t j,
    scalar_t* cache_t_ptr,
    scalar_t* cache_h_ptr,
    scalar_t* cache_w_ptr,
    int64_t mrope_section_t,
    int64_t mrope_section_h,
    int64_t mrope_section_w,
    bool mrope_interleaved) {
  if (mrope_interleaved) {
    if (j % 3 == 1 && j <= mrope_section_h * 3) return cache_h_ptr;
    if (j % 3 == 2 && j <= mrope_section_w * 3) return cache_w_ptr;
    return cache_t_ptr;
  }
  if (j < mrope_section_t) return cache_t_ptr;
  if (j < mrope_section_t + mrope_section_h) return cache_h_ptr;
  return cache_w_ptr;
}

template <typename scalar_t>
void multimodal_rotary_embedding_neox_2D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t key_stride_s,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_tokens,
    int64_t mrope_section_t,
    int64_t mrope_section_h,
    int64_t mrope_section_w,
    int64_t positions_stride0,
    bool mrope_interleaved) {
  int64_t embed_dim = rotary_dim / 2;
  auto compute_loop =
      [&](int64_t token_head, scalar_t* cache_t_ptr, scalar_t* cache_h_ptr, scalar_t* cache_w_ptr, scalar_t* qk) {
        for (int64_t j = 0; j < embed_dim; ++j) {
          int64_t x_index = j;
          int64_t y_index = embed_dim + j;

          int64_t out_x = token_head + x_index;
          int64_t out_y = token_head + y_index;

          scalar_t* cache_ptr = get_cache_ptr(
              j,
              cache_t_ptr,
              cache_h_ptr,
              cache_w_ptr,
              mrope_section_t,
              mrope_section_h,
              mrope_section_w,
              mrope_interleaved);
          float _cos = cache_ptr[x_index];
          float _sin = cache_ptr[y_index];

          float _q_x = qk[out_x];
          float _q_y = qk[out_y];

          qk[out_x] = _q_x * _cos - _q_y * _sin;
          qk[out_y] = _q_y * _cos + _q_x * _sin;
        }
      };
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    int64_t token_idx = {0};
    data_index_init(begin, token_idx, num_tokens);
    for (int i = begin; i < end; ++i) {
      int64_t pos_t = positions[token_idx];
      int64_t pos_h = positions[positions_stride0 + token_idx];
      int64_t pos_w = positions[positions_stride0 * 2 + token_idx];
      scalar_t* cache_t_ptr = cos_sin_cache + pos_t * rotary_dim;
      scalar_t* cache_h_ptr = cos_sin_cache + pos_h * rotary_dim;
      scalar_t* cache_w_ptr = cos_sin_cache + pos_w * rotary_dim;

      for (int64_t i = 0; i < num_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = token_idx * query_stride_s + head_idx * head_size;
        compute_loop(token_head, cache_t_ptr, cache_h_ptr, cache_w_ptr, query);
      }

      for (int64_t i = 0; i < num_kv_heads; ++i) {
        int64_t head_idx = i;
        int64_t token_head = token_idx * key_stride_s + head_idx * head_size;
        compute_loop(token_head, cache_t_ptr, cache_h_ptr, cache_w_ptr, key);
      }
      data_index_step(token_idx, num_tokens);
    }
  });
}

template <typename scalar_t>
void rotary_embedding_4D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_b,
    int64_t query_stride_s,
    int64_t query_stride_h,
    int64_t key_stride_b,
    int64_t key_stride_s,
    int64_t key_stride_h,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t batch_size,
    int64_t seq_len) {
  int64_t embed_dim = rotary_dim / 2;

  at::parallel_for(0, batch_size * seq_len * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t bs = {0}, seq = {0}, i = {0};
    data_index_init(begin, bs, batch_size, seq, seq_len, i, num_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t head_idx = i;
      int64_t token_head = bs * query_stride_b + seq * query_stride_s + head_idx * query_stride_h;
      scalar_t* head_query = token_head + query;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t rot_offset = j;
        int64_t x_index = 2 * rot_offset;
        int64_t y_index = 2 * rot_offset + 1;

        float cos = cos_cache_ptr[rot_offset];
        float sin = sin_cache_ptr[rot_offset];

        float x = head_query[x_index];
        float y = head_query[y_index];

        head_query[x_index] = x * cos - y * sin;
        head_query[y_index] = y * cos + x * sin;
      }
      data_index_step(bs, batch_size, seq, seq_len, i, num_heads);
    }
  });

  at::parallel_for(0, batch_size * seq_len * num_kv_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t bs = {0}, seq = {0}, i = {0};
    data_index_init(begin, bs, batch_size, seq, seq_len, i, num_kv_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos = positions[bs * seq_len + seq];
      scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
      scalar_t* cos_cache_ptr = cache_ptr;
      scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      int64_t head_idx = i;
      int64_t token_head = bs * key_stride_b + seq * key_stride_s + head_idx * head_size;
      scalar_t* head_key = key + token_head;
      for (int64_t j = 0; j < embed_dim; j += 1) {
        int64_t rot_offset = j;
        int64_t x_index = 2 * rot_offset;
        int64_t y_index = 2 * rot_offset + 1;

        float cos = cos_cache_ptr[rot_offset];
        float sin = sin_cache_ptr[rot_offset];

        float x = head_key[x_index];
        float y = head_key[y_index];

        head_key[x_index] = x * cos - y * sin;
        head_key[y_index] = y * cos + x * sin;
      }
      data_index_step(bs, batch_size, seq, seq_len, i, num_kv_heads);
    }
  });
}

template <typename scalar_t>
void multimodal_rotary_embedding_2D_kernel_impl(
    int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    scalar_t* __restrict__ cos_sin_cache,
    int64_t rotary_dim,
    int64_t query_stride_s,
    int64_t key_stride_s,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_tokens,
    int64_t mrope_section_t,
    int64_t mrope_section_h,
    int64_t mrope_section_w,
    int64_t positions_stride0,
    bool mrope_interleaved) {
  int64_t embed_dim = rotary_dim / 2;
  auto compute_loop = [&](scalar_t* cache_t_ptr, scalar_t* cache_h_ptr, scalar_t* cache_w_ptr, scalar_t* head_query) {
    for (int64_t j = 0; j < embed_dim; j += 1) {
      int64_t rot_offset = j;
      int64_t x_index = 2 * rot_offset;
      int64_t y_index = 2 * rot_offset + 1;

      scalar_t* cache_ptr = get_cache_ptr(
          j,
          cache_t_ptr,
          cache_h_ptr,
          cache_w_ptr,
          mrope_section_t,
          mrope_section_h,
          mrope_section_w,
          mrope_interleaved);
      float cos = cache_ptr[rot_offset];
      float sin = cache_ptr[rot_offset + embed_dim];

      float x = head_query[x_index];
      float y = head_query[y_index];

      head_query[x_index] = x * cos - y * sin;
      head_query[y_index] = y * cos + x * sin;
    }
  };
  at::parallel_for(0, num_tokens * num_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t token_idx = {0}, i = {0};
    data_index_init(begin, token_idx, num_tokens, i, num_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos_t = positions[token_idx];
      int64_t pos_h = positions[positions_stride0 + token_idx];
      int64_t pos_w = positions[positions_stride0 * 2 + token_idx];
      scalar_t* cache_t_ptr = cos_sin_cache + pos_t * rotary_dim;
      scalar_t* cache_h_ptr = cos_sin_cache + pos_h * rotary_dim;
      scalar_t* cache_w_ptr = cos_sin_cache + pos_w * rotary_dim;
      int64_t head_idx = i;
      int64_t token_head = token_idx * query_stride_s + head_idx * head_size;
      scalar_t* head_query = token_head + query;
      compute_loop(cache_t_ptr, cache_h_ptr, cache_w_ptr, head_query);
      data_index_step(token_idx, num_tokens, i, num_heads);
    }
  });

  at::parallel_for(0, num_tokens * num_kv_heads, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t token_idx{0}, i = {0};
    data_index_init(begin, token_idx, num_tokens, i, num_kv_heads);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t pos_t = positions[token_idx];
      int64_t pos_h = positions[positions_stride0 + token_idx];
      int64_t pos_w = positions[positions_stride0 * 2 + token_idx];
      scalar_t* cache_t_ptr = cos_sin_cache + pos_t * rotary_dim;
      scalar_t* cache_h_ptr = cos_sin_cache + pos_h * rotary_dim;
      scalar_t* cache_w_ptr = cos_sin_cache + pos_w * rotary_dim;
      int64_t head_idx = i;
      int64_t token_head = token_idx * key_stride_s + head_idx * head_size;
      scalar_t* head_key = key + token_head;
      compute_loop(cache_t_ptr, cache_h_ptr, cache_w_ptr, head_key);
      data_index_step(token_idx, num_tokens, i, num_kv_heads);
    }
  });
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox) {
  CHECK_DIM(1, positions);
  const auto input_dim = query.dim();
  const auto input_dtype = query.scalar_type();
  TORCH_CHECK(
      input_dim == 2 || input_dim == 3 || input_dim == 4,
      " Query/Key must be 2D [num_tokens, num_heads*head_size] or 3D [num_tokens, num_heads, head_size] or 4D "
      "[batch_size, seq_len, num_heads, head_size] tensor");
  CHECK_DIM(2, cos_sin_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);

  int64_t rotary_dim = cos_sin_cache.size(1);
  if (input_dim == 3) {
    // TODO: add support for head_dim != rotary_dim case when input_dim=3
    CHECK_EQ(query.size(-1), rotary_dim);
    // TODO: add support for kv_head != 1
    CHECK_EQ(key.size(1), 1);
  }

  int64_t num_tokens = positions.numel();
  if (input_dim <= 3) {
    CHECK_EQ(key.size(0), num_tokens);
    CHECK_EQ(query.size(0), num_tokens);
  }

  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t num_heads = input_dim == 2 ? query.size(-1) / head_size : query.size(-2);
  int64_t num_kv_heads = input_dim == 2 ? key.size(-1) / head_size : key.size(-2);
  int64_t key_stride_s = key.stride(0);
  int64_t query_stride_s = query.stride(0);

  int64_t query_stride_h = input_dim == 2 ? head_size : query.stride(-2);
  int64_t key_stride_h = input_dim == 2 ? head_size : key.stride(-2);
  at::Tensor query_out = at::empty_like(query);
  at::Tensor key_out = at::empty_like(key);
  int64_t query_out_stride_s = query_out.stride(0);
  int64_t key_out_stride_s = key_out.stride(0);
  // output stride of num head dim is meaningful only when input dim = 3
  int64_t query_out_stride_h = input_dim == 3 ? query_out.stride(1) : -1;
  int64_t batch_size = 1;
  int64_t seq_len = num_tokens;
  int64_t query_stride_b = 0;
  int64_t key_stride_b = 0;
  if (input_dim == 4) {
    batch_size = query.size(0);
    seq_len = query.size(1);
    query_stride_b = query.stride(0);
    key_stride_b = key.stride(0);
    query_stride_s = query.stride(1);
    key_stride_s = key.stride(1);
    CHECK_EQ(batch_size, key.size(0));
    CHECK_EQ(seq_len, key.size(1));
    CHECK_EQ(key.size(0) * key.size(1), num_tokens);
    CHECK_EQ(query.size(0) * query.size(1), num_tokens);
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
    if (input_dim == 2 || input_dim == 4) {
      if (is_neox) {
        rotary_embedding_neox_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      } else {
        rotary_embedding_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_b,
            query_stride_s,
            query_stride_h,
            key_stride_b,
            key_stride_s,
            key_stride_h,
            num_heads,
            num_kv_heads,
            head_size,
            batch_size,
            seq_len);
      }
      query_out = query;
      key_out = key;

    } else {
      TORCH_CHECK(
          is_neox == false, " Query/Key with 3D [num_tokens, num_heads, head_size] does not support neox rope yet");
      // TODO: add neox style support for rope impl with 3D inputs
      rotary_embedding_3D_kernel_impl<scalar_t>(
          query_out.data_ptr<scalar_t>(),
          key_out.data_ptr<scalar_t>(),
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          num_tokens,
          num_heads,
          num_kv_heads,
          head_size,
          rotary_dim,
          query_stride_s,
          query_out_stride_s,
          key_out_stride_s,
          key_stride_s,
          query_stride_h,
          query_out_stride_h);
    }
  });
  return std::make_tuple(query_out, key_out);
}

// Matches the semantics of apply_rotary_pos_emb_native_eager (neox style, computed in
// float, out-of-place):
// query: [..., num_heads, head_size]
// key: [..., num_kv_heads, head_size]
// cos/sin: query/key with the num_heads dim (at unsqueeze_dim) removed, e.g.
//   3D: query [num_tokens, num_heads, head_size], cos [num_tokens, head_size] (unsqueeze_dim=1)
//   4D: query [batch, num_heads, seq_len, head_size], cos [batch, seq_len, head_size] (unsqueeze_dim=1)
std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb_cpu(
    at::Tensor& query, at::Tensor& key, at::Tensor& cos, at::Tensor& sin, int64_t unsqueeze_dim) {
  const int64_t ndim = query.dim();
  TORCH_CHECK(ndim >= 3, "query/key must be at least 3D [..., num_heads, head_size]");
  CHECK_EQ(key.dim(), ndim);
  CHECK_EQ(cos.dim(), ndim - 1);
  CHECK_EQ(sin.dim(), ndim - 1);
  TORCH_CHECK(unsqueeze_dim >= 0 && unsqueeze_dim < ndim - 1, "invalid unsqueeze_dim ", unsqueeze_dim);
  const auto input_dtype = query.scalar_type();
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(), "cos and sin must have the same data type");

  const int64_t head_size = query.size(-1);
  TORCH_CHECK(head_size % 2 == 0, "head_size must be even");
  CHECK_EQ(head_size, key.size(-1));
  CHECK_EQ(head_size, cos.size(-1));
  CHECK_EQ(head_size, sin.size(-1));

  // verify broadcast compatibility and fold the non-head dims into outer/inner sizes
  int64_t outer_size = 1;
  int64_t inner_size = 1;
  for (int64_t d = 0; d < ndim - 1; ++d) {
    if (d == unsqueeze_dim) {
      continue;
    }
    const int64_t cos_d = d < unsqueeze_dim ? d : d - 1;
    CHECK_EQ(query.size(d), key.size(d));
    CHECK_EQ(query.size(d), cos.size(cos_d));
    CHECK_EQ(query.size(d), sin.size(cos_d));
    if (d < unsqueeze_dim) {
      outer_size *= query.size(d);
    } else {
      inner_size *= query.size(d);
    }
  }
  const int64_t num_heads = query.size(unsqueeze_dim);
  const int64_t num_kv_heads = key.size(unsqueeze_dim);

  at::Tensor query_c = query.contiguous();
  at::Tensor key_c = key.contiguous();
  at::Tensor cos_c = cos.contiguous();
  at::Tensor sin_c = sin.contiguous();
  // match the native impl, which always computes with float cos/sin
  if (cos_c.scalar_type() != at::kFloat && cos_c.scalar_type() != input_dtype) {
    cos_c = cos_c.to(at::kFloat);
    sin_c = sin_c.to(at::kFloat);
  }
  at::Tensor query_out = at::empty_like(query_c);
  at::Tensor key_out = at::empty_like(key_c);

  AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::kFloat, input_dtype, "apply_rotary_pos_emb_cpu", [&] {
    if (cos_c.scalar_type() == at::kFloat) {
      apply_rotary_pos_emb_kernel_impl<scalar_t, float>(
          query_out.data_ptr<scalar_t>(),
          query_c.data_ptr<scalar_t>(),
          cos_c.data_ptr<float>(),
          sin_c.data_ptr<float>(),
          outer_size,
          num_heads,
          inner_size,
          head_size);
      apply_rotary_pos_emb_kernel_impl<scalar_t, float>(
          key_out.data_ptr<scalar_t>(),
          key_c.data_ptr<scalar_t>(),
          cos_c.data_ptr<float>(),
          sin_c.data_ptr<float>(),
          outer_size,
          num_kv_heads,
          inner_size,
          head_size);
    } else {
      apply_rotary_pos_emb_kernel_impl<scalar_t, scalar_t>(
          query_out.data_ptr<scalar_t>(),
          query_c.data_ptr<scalar_t>(),
          cos_c.data_ptr<scalar_t>(),
          sin_c.data_ptr<scalar_t>(),
          outer_size,
          num_heads,
          inner_size,
          head_size);
      apply_rotary_pos_emb_kernel_impl<scalar_t, scalar_t>(
          key_out.data_ptr<scalar_t>(),
          key_c.data_ptr<scalar_t>(),
          cos_c.data_ptr<scalar_t>(),
          sin_c.data_ptr<scalar_t>(),
          outer_size,
          num_kv_heads,
          inner_size,
          head_size);
    }
  });
  return std::make_tuple(query_out, key_out);
}

// positions: [num_tokens] (text only) or [3, num_tokens] (T/H/W positions with multimodal inputs)
// query: [num_tokens, num_heads * head_size]
// key: [num_tokens, num_kv_heads * head_size]
// cos_sin_cache: [max_position_embeddings, rotary_dim]
// mrope_section: [t, h, w]
std::tuple<at::Tensor, at::Tensor> multimodal_rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    const std::optional<std::vector<int64_t>>& mrope_section,
    bool mrope_interleaved,
    bool is_neox) {
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2, "positions must be a 1D or 2D tensor");
  CHECK_DIM(2, query);
  CHECK_DIM(2, key);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  int64_t rotary_dim = cos_sin_cache.size(1);
  int64_t num_tokens = positions.size(-1);
  CHECK_EQ(key.size(0), num_tokens);
  CHECK_EQ(query.size(0), num_tokens);
  const auto input_dtype = query.scalar_type();
  TORCH_CHECK(positions.scalar_type() == at::kLong, "expect positions to be int64, got ", positions.scalar_type());
  TORCH_CHECK(input_dtype == key.scalar_type(), "query and key must have the same data type");
  TORCH_CHECK(input_dtype == cos_sin_cache.scalar_type(), "query and cos_sin_cache must have the same data type");

  int64_t num_heads = query.size(-1) / head_size;
  int64_t num_kv_heads = key.size(-1) / head_size;
  int64_t key_stride_s = key.stride(0);
  int64_t query_stride_s = query.stride(0);

  if (positions.dim() == 2) {
    TORCH_CHECK(mrope_section.has_value(), "mrope_section must be provided when positions is 2D");
    auto mrope_section_val = mrope_section.value();
    CHECK_EQ(mrope_section_val.size(), 3);
    CHECK_EQ(positions.size(0), 3);
    int64_t mrope_section_t = mrope_section_val[0];
    int64_t mrope_section_h = mrope_section_val[1];
    int64_t mrope_section_w = mrope_section_val[2];
    int64_t positions_stride0 = positions.stride(0);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
      if (is_neox) {
        multimodal_rotary_embedding_neox_2D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_s,
            key_stride_s,
            num_heads,
            num_kv_heads,
            head_size,
            num_tokens,
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            positions_stride0,
            mrope_interleaved);
      } else {
        multimodal_rotary_embedding_2D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            query_stride_s,
            key_stride_s,
            num_heads,
            num_kv_heads,
            head_size,
            num_tokens,
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            positions_stride0,
            mrope_interleaved);
      }
    });
  } else {  // positions.dim() == 1
    AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_embedding_cpu", [&] {
      if (is_neox) {
        rotary_embedding_neox_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            0,
            query_stride_s,
            head_size,
            0,
            key_stride_s,
            head_size,
            num_heads,
            num_kv_heads,
            head_size,
            1,
            num_tokens);
      } else {
        rotary_embedding_4D_kernel_impl<scalar_t>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            rotary_dim,
            0,
            query_stride_s,
            head_size,
            0,
            key_stride_s,
            head_size,
            num_heads,
            num_kv_heads,
            head_size,
            1,
            num_tokens);
      }
    });
  }
  return std::make_tuple(query, key);
}
