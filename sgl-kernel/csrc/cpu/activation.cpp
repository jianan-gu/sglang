#include "common.h"
#include "vec.h"
#include "vec_vllm.h"

namespace {

template <typename scalar_t, typename func_t, typename vec_func_t>
void act_and_mul_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t num_tokens, int64_t dim,
    const func_t& f,
    const vec_func_t& vf) {

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t kVecSize = bVec::size();
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      // local ptrs
      const scalar_t* __restrict__ input_ptr = input + i * 2 * dim;
      const scalar_t* __restrict__ input_other_ptr = input_ptr + dim;
      scalar_t* __restrict__ output_ptr = output + i * dim;

      int64_t d;
      #pragma GCC unroll 4
      for (d = 0; d <= dim - kVecSize; d += kVecSize) {
        bVec x_bvec = bVec::loadu(input_ptr + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        bVec y_bvec = bVec::loadu(input_other_ptr + d);
        fVec y_fvec0, y_fvec1;
        std::tie(y_fvec0, y_fvec1) = at::vec::convert_to_float(y_bvec);

        x_fvec0 = vf(x_fvec0);
        x_fvec1 = vf(x_fvec1);

        x_fvec0 = x_fvec0 * y_fvec0;
        x_fvec1 = x_fvec1 * y_fvec1;

        x_bvec = convert_from_float_ext<scalar_t>(x_fvec0, x_fvec1);
        x_bvec.store(output_ptr + d);
      }
      #pragma GCC unroll 4
      for (; d < dim; ++d) {
        float x_val = static_cast<float>(input_ptr[d]);
        float y_val = static_cast<float>(input_other_ptr[d]);
        output_ptr[d] = f(x_val) * y_val;
      }
    }
  });
}

template <typename scalar_t>
void rotary_embedding_impl(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           /// [batch_size, seq_len, num_heads,
                                   /// head_size] or [num_tokens, num_heads,
                                   /// head_size]
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();

  const int embed_dim = rot_dim / 2;
  bool flag = (embed_dim % VEC_ELEM_NUM == 0);
  const int loop_upper = flag ? embed_dim : embed_dim - VEC_ELEM_NUM;

  auto compute_loop = [&](const int64_t token_head, const scalar_t* cache_ptr,
                          scalar_t* qk) {
    int j = 0;
    for (; j < loop_upper; j += VEC_ELEM_NUM) {
      const int rot_offset = j;
      const int x_index = rot_offset;
      const int y_index = embed_dim + rot_offset;

      const int64_t out_x = token_head + x_index;
      const int64_t out_y = token_head + y_index;

      const scalar_vec_t cos(cache_ptr + x_index);
      const scalar_vec_t sin(cache_ptr + y_index);

      const scalar_vec_t q_x(qk + out_x);
      const scalar_vec_t q_y(qk + out_y);

      vec_op::FP32Vec8 fp32_cos(cos);
      vec_op::FP32Vec8 fp32_sin(sin);

      vec_op::FP32Vec8 fp32_q_x(q_x);
      vec_op::FP32Vec8 fp32_q_y(q_y);

      auto out1 = fp32_q_x * fp32_cos - fp32_q_y * fp32_sin;
      scalar_vec_t(out1).save(qk + out_x);

      auto out2 = fp32_q_y * fp32_cos + fp32_q_x * fp32_sin;
      scalar_vec_t(out2).save(qk + out_y);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        const int x_index = j;
        const int y_index = embed_dim + j;

        const int64_t out_x = token_head + x_index;
        const int64_t out_y = token_head + y_index;

        const float fp32_cos = cache_ptr[x_index];
        const float fp32_sin = cache_ptr[y_index];

        const float fp32_q_x = qk[out_x];
        const float fp32_q_y = qk[out_y];

        qk[out_x] = fp32_q_x * fp32_cos - fp32_q_y * fp32_sin;
        qk[out_y] = fp32_q_y * fp32_cos + fp32_q_x * fp32_sin;
      }
    }
  };

#pragma omp parallel for
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    for (int i = 0; i < num_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head =
          token_idx * query_stride + head_idx * head_size;
      compute_loop(token_head, cache_ptr, query);
    }

    for (int i = 0; i < num_kv_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head = token_idx * key_stride + head_idx * head_size;
      compute_loop(token_head, cache_ptr, key);
    }
  }
}

template <typename scalar_t>
void rotary_embedding_gptj_impl(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           /// [batch_size, seq_len, num_heads,
                                   /// head_size] or [num_tokens, num_heads,
                                   /// head_size]
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size,
    const int num_tokens) {
  const int embed_dim = rot_dim / 2;

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_heads; ++i) {
      int64_t pos = positions[token_idx];
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const scalar_t* cos_cache_ptr = cache_ptr;
      const scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head =
          token_idx * query_stride + head_idx * head_size;
      scalar_t* head_query = token_head + query;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = cos_cache_ptr[rot_offset];
        const float sin = sin_cache_ptr[rot_offset];

        const float x = head_query[x_index];
        const float y = head_query[y_index];

        head_query[x_index] = x * cos - y * sin;
        head_query[y_index] = y * cos + x * sin;
      }
    }
  }

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_kv_heads; ++i) {
      int64_t pos = positions[token_idx];
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const scalar_t* cos_cache_ptr = cache_ptr;
      const scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head = token_idx * key_stride + head_idx * head_size;
      scalar_t* head_key = key + token_head;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = cos_cache_ptr[rot_offset];
        const float sin = sin_cache_ptr[rot_offset];

        const float x = head_key[x_index];
        const float y = head_key[y_index];

        head_key[x_index] = x * cos - y * sin;
        head_key[y_index] = y * cos + x * sin;
      }
    }
  }
}

} // anonymous namespace

// input   : {num_tokens, 2 * d}
// output  : {num_tokens, d}
at::Tensor silu_and_mul_cpu(at::Tensor& input) {
  RECORD_FUNCTION("sgl-kernel::silu_and_mul_cpu", std::vector<c10::IValue>({input}));
  auto sizes = input.sizes().vec();
  int64_t last_dim = input.ndimension() - 1;
  int64_t d = sizes[last_dim] / 2;
  sizes[last_dim] = d;
  int64_t num_tokens = input.numel() / input.size(-1);
  at::Tensor out = at::empty(sizes, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    using Vec = at::vec::Vectorized<float>;
    act_and_mul_kernel_impl(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        num_tokens,
        d,
        [](float x) { return x / (1.f + std::exp(-x)); },
        [](Vec x) { return x / (Vec(1.f) + x.neg().exp()); });
  });
  return out;
}

void rotary_embedding_cpu(at::Tensor& positions, at::Tensor& query,
  at::Tensor& key, int64_t head_size,
  at::Tensor& cos_sin_cache, bool is_neox) {
int num_tokens = positions.numel();
int rot_dim = cos_sin_cache.size(1);
int num_heads = query.size(-1) / head_size;
int num_kv_heads = key.size(-1) / head_size;
int64_t key_stride = key.stride(-2);
int64_t query_stride = query.stride(-2);

AT_DISPATCH_REDUCED_FLOATING_TYPES(
query.scalar_type(), "rotary_embedding_impl", [&] {
if (is_neox) {
rotary_embedding_impl(
positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
head_size, num_tokens);
} else {
rotary_embedding_gptj_impl(
positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
head_size, num_tokens);
}

});
}