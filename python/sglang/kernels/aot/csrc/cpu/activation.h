#pragma once

#include "vec.h"

template <typename scalar_t, typename input_t>
inline void clamped_silu_and_mul_stub(
    scalar_t* __restrict__ out,
    const input_t* __restrict__ input,   // gate buffer
    const input_t* __restrict__ input2,  // up buffer
    int64_t size,
    const float limit) {
  static_assert(
      std::is_same_v<input_t, float> || std::is_same_v<input_t, scalar_t>,
      "clamped_silu_and_mul_stub only supports input_t == float or input_t == scalar_t");
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  const fVec limit_v = fVec(limit);
  const fVec nlimit_v = fVec(-limit);

  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - bVec::size(); d += bVec::size()) {
    auto [x0, x1] = load_float_vec2(input + d);
    auto [y0, y1] = load_float_vec2(input2 + d);

    // gate: clamp at upper bound only; up: clamp symmetrically
    x0 = at::vec::minimum(x0, limit_v);
    x1 = at::vec::minimum(x1, limit_v);
    y0 = at::vec::minimum(limit_v, at::vec::maximum(nlimit_v, y0));
    y1 = at::vec::minimum(limit_v, at::vec::maximum(nlimit_v, y1));
    // silu(gate) * up
    x0 = x0 / (one + x0.neg().exp_u20());
    x1 = x1 / (one + x1.neg().exp_u20());
    x0 = x0 * y0;
    x1 = x1 * y1;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
#pragma GCC unroll 4
  for (; d < size; ++d) {
    float gate = std::min(static_cast<float>(input[d]), limit);
    float up = std::clamp(static_cast<float>(input2[d]), -limit, limit);
    out[d] = static_cast<scalar_t>(gate / (1.f + std::exp(-gate)) * up);
  }
}
