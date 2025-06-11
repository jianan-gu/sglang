#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {
  #define QUANT_A_THRESHOLD 30720
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

  template <typename scalar_t>
inline void copy_add_stub(scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) + fVec::loadu(bias + d);
    fVec data1 = fVec::loadu(input + d + fVec::size()) + fVec::loadu(bias + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + bias[d]);
  }
}

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
    const scalar_t* __restrict__ A, const at::quint4x2* __restrict__ B, scalar_t* __restrict__ C,
    const uint8_t* __restrict__ Bz, const scalar_t* __restrict__ Bs,
    const float* __restrict__ bias, int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t strideBz, int64_t strideBs) {
  TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};


inline std::array<__m256i, 2> load_zps_4vnni(int8_t* zps) {
  // broadcast 01234567 to
  // 01234567012345670123456701234567
  __m256i vzps_low = _mm256_set1_epi64x(*reinterpret_cast<long*>(zps));
  __m256i vzps_high = _mm256_set1_epi64x(*reinterpret_cast<long*>(zps + 8));
  // shuffle from
  // 01234567012345670123456701234567
  // to
  // 00001111222233334444555566667777
  __m256i shuffle_mask = _mm256_set_epi8(
      7,
      7,
      7,
      7,
      6,
      6,
      6,
      6,
      5,
      5,
      5,
      5,
      4,
      4,
      4,
      4,
      3,
      3,
      3,
      3,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0);
  vzps_low = _mm256_shuffle_epi8(vzps_low, shuffle_mask);
  vzps_high = _mm256_shuffle_epi8(vzps_high, shuffle_mask);
  return {vzps_low, vzps_high};
}

// template <long ldb, int qw_type, bool sym_quant_w>
// struct Dequantize<
//     int8_t,
//     ldb,
//     /*N_GROUP_SIZE*/ 16,
//     qw_type,
//     sym_quant_w,
//     /*use_g_idx*/ false> {
//   template <int quant_a_mode> 
//   static inline void call(
//       uint8_t* qB,
//       long K,
//       long N,
//       int8_t* zps,
//       int8_t* B,
//       int32_t* compensation) {
//   TORCH_CHECK(false, "not implemented");
//   }
// };
template <
    typename Tin,
    long ldb,
    long N_GROUP_SIZE,
    int qw_type,
    bool sym_quant_w,
    bool use_g_idx>
struct Dequantize {
  static void call(
      uint8_t* qB,
      long K,
      long N,
      Tin* scales,
      Tin* zps,
      Tin* B,
      int k_start = 0,
      int* g_idx = nullptr);
};
// template <typename T>
// at::Tensor quantize_per_tensor(
//     const at::Tensor& t,
//     float scale,
//     int32_t zp,
//     bool is_sym_quant) {
//   // TODO(jgong5): optimize me
//   auto t_q = is_sym_quant ? t / scale : t / scale + zp;
//   t_q = is_sym_quant ? at::clamp(at::round(t_q), -128, 127)
//                      : at::clamp(at::round(t_q), 0, 255);
//   return is_sym_quant ? t_q.to(at::kChar) : t_q.to(at::kByte);
// }
#if defined(CPU_CAPABILITY_AVX512)

template <typename T>
at::Tensor quantize_per_tensor(
    const at::Tensor& t,
    float scale,
    int32_t zp,
    bool is_sym_quant) {
  // TODO(jgong5): optimize me
  auto t_q = is_sym_quant ? t / scale : t / scale + zp;
  t_q = is_sym_quant ? at::clamp(at::round(t_q), -128, 127)
                     : at::clamp(at::round(t_q), 0, 255);
  return is_sym_quant ? t_q.to(at::kChar) : t_q.to(at::kByte);
}

template <>
inline at::Tensor quantize_per_tensor<float>(
    const at::Tensor& t,
    float scale,
    int32_t zp,
    bool is_sym_quant) {
#if defined(CPU_CAPABILITY_AVX512)
  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  auto in_ptr0 = t.data_ptr<float>();
  uint8_t* out_ptr0 = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr0 = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize, scale, zp](
          float* in_ptr, int start, int end, uint8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr + i1, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp2 = tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(0.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(255.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<uint8_t>(tmp9);
          tmp10.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = tmp0 / static_cast<float>(scale);
          auto tmp2 = tmp1 + static_cast<float>(zp);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(0.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(255.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<unsigned char>(tmp9);
          out_ptr[i1] = tmp10;
        }
      };
  auto quantize_block_sym =
      [vecsize, scale, zp](float* in_ptr, int start, int end, int8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr + i1, vecsize);
          auto tmp1 =
              tmp0 / at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp2 = tmp1 + at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp3 = tmp2.round();
          auto tmp4 = (tmp3);
          auto tmp5 = at::vec::Vectorized<float>(static_cast<float>(-128.0));
          auto tmp6 = at::vec::maximum(tmp4, tmp5);
          auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(127.0));
          auto tmp8 = at::vec::minimum(tmp6, tmp7);
          auto tmp9 = (tmp8);
          auto tmp10 = at::vec::convert_float_to_int8<int8_t>(tmp9);
          tmp10.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = tmp0 / static_cast<float>(scale);
          auto tmp2 = tmp1 + static_cast<float>(zp);
          auto tmp3 = std::nearbyint(tmp2);
          auto tmp4 = static_cast<float>(tmp3);
          auto tmp5 = static_cast<float>(-128.0);
          auto tmp6 = 0;
          if (at::_isnan(tmp4)) {
            tmp6 = tmp4;
          }
          tmp6 = tmp4 > tmp5 ? tmp4 : tmp5;
          auto tmp7 = static_cast<float>(127.0);
          auto tmp8 = 0;
          if (at::_isnan(tmp6)) {
            tmp8 = tmp6;
          }
          tmp8 = tmp6 < tmp7 ? tmp6 : tmp7;
          auto tmp9 = static_cast<float>(tmp8);
          auto tmp10 = static_cast<int8_t>(tmp9);
          out_ptr[i1] = tmp10;
        }
      };
  if (n > QUANT_A_THRESHOLD) {
    int num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      if (is_sym_quant) {
        quantize_block_sym(in_ptr0, vec_start, vec_end, out_sym_ptr0);
      } else {
        quantize_block(in_ptr0, vec_start, vec_end, out_ptr0);
      }
    }
  } else {
    if (is_sym_quant) {
      quantize_block_sym(in_ptr0, 0, n, out_sym_ptr0);
    } else {
      quantize_block(in_ptr0, 0, n, out_ptr0);
    }
  }
  return out;
#else
  return at::quantize_per_tensor(t, scale, zp, c10::kQUInt8);
#endif
}

template <>
inline at::Tensor quantize_per_tensor<at::BFloat16>(
    const at::Tensor& t,
    float scale,
    int32_t zp,
    bool is_sym_quant) {
#if defined(CPU_CAPABILITY_AVX512)
  auto out_dtype = is_sym_quant ? at::kChar : at::kByte;
  at::Tensor out = at::empty_like(t, out_dtype);
  auto in_ptr0 = t.data_ptr<at::BFloat16>();
  uint8_t* out_ptr0 = is_sym_quant ? nullptr : out.data_ptr<uint8_t>();
  int8_t* out_sym_ptr0 = is_sym_quant ? out.data_ptr<int8_t>() : nullptr;
  auto n = t.numel();
  auto K = t.size(-1);
  auto M = t.numel() / K;
  auto vecsize = at::vec::Vectorized<float>::size();
  auto quantize_block =
      [vecsize, scale, zp](
          at::BFloat16* in_ptr, int start, int end, uint8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 =
              at::vec::Vectorized<at::BFloat16>::loadu(in_ptr + i1, vecsize);
          at::vec::Vectorized<float> res_vec1(0);
          at::vec::Vectorized<float> res_vec2(0);
          std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
          auto tmp1 = res_vec1;
          auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = tmp5.round();
          auto tmp7 = (tmp6);
          auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(0.0));
          auto tmp9 = at::vec::maximum(tmp7, tmp8);
          auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(255.0));
          auto tmp11 = at::vec::minimum(tmp9, tmp10);
          auto tmp12 = (tmp11);
          auto tmp13 = at::vec::convert_float_to_int8<uint8_t>(tmp12);
          tmp13.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = static_cast<float>(tmp0);
          auto tmp2 = static_cast<float>(scale);
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = static_cast<float>(zp);
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = std::nearbyint(tmp5);
          auto tmp7 = static_cast<float>(tmp6);
          auto tmp8 = static_cast<float>(0.0);
          auto tmp9 = 0;
          if (at::_isnan(tmp7)) {
            tmp9 = tmp7;
          }
          tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
          auto tmp10 = static_cast<float>(255.0);
          auto tmp11 = 0;
          if (at::_isnan(tmp9)) {
            tmp11 = tmp9;
          }
          tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
          auto tmp12 = static_cast<float>(tmp11);
          auto tmp13 = static_cast<unsigned char>(tmp12);
          out_ptr[i1] = tmp13;
        }
      };
  auto quantize_block_sym =
      [vecsize, scale, zp](
          at::BFloat16* in_ptr, int start, int end, int8_t* out_ptr) {
        int i1;
        for (i1 = start; i1 < end / vecsize * vecsize; i1 += vecsize) {
          auto tmp0 =
              at::vec::Vectorized<at::BFloat16>::loadu(in_ptr + i1, vecsize);
          at::vec::Vectorized<float> res_vec1(0);
          at::vec::Vectorized<float> res_vec2(0);
          std::tie(res_vec1, res_vec2) = at::vec::convert_bfloat16_float(tmp0);
          auto tmp1 = res_vec1;
          auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(scale));
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = at::vec::Vectorized<float>(static_cast<float>(zp));
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = tmp5.round();
          auto tmp7 = (tmp6);
          auto tmp8 = at::vec::Vectorized<float>(static_cast<float>(-128.0));
          auto tmp9 = at::vec::maximum(tmp7, tmp8);
          auto tmp10 = at::vec::Vectorized<float>(static_cast<float>(127.0));
          auto tmp11 = at::vec::minimum(tmp9, tmp10);
          auto tmp12 = (tmp11);
          auto tmp13 = at::vec::convert_float_to_int8<int8_t>(tmp12);
          tmp13.store(out_ptr + i1, vecsize);
        }
        for (; i1 < end; i1++) {
          auto tmp0 = in_ptr[i1];
          auto tmp1 = static_cast<float>(tmp0);
          auto tmp2 = static_cast<float>(scale);
          auto tmp3 = tmp1 / tmp2;
          auto tmp4 = static_cast<float>(zp);
          auto tmp5 = tmp3 + tmp4;
          auto tmp6 = std::nearbyint(tmp5);
          auto tmp7 = static_cast<float>(tmp6);
          auto tmp8 = static_cast<float>(-128.0);
          auto tmp9 = 0;
          if (at::_isnan(tmp7)) {
            tmp9 = tmp7;
          }
          tmp9 = tmp7 > tmp8 ? tmp7 : tmp8;
          auto tmp10 = static_cast<float>(127.0);
          auto tmp11 = 0;
          if (at::_isnan(tmp9)) {
            tmp11 = tmp9;
          }
          tmp11 = tmp9 < tmp10 ? tmp9 : tmp10;
          auto tmp12 = static_cast<float>(tmp11);
          auto tmp13 = static_cast<int8_t>(tmp12);
          out_ptr[i1] = tmp13;
        }
      };
  if (n > QUANT_A_THRESHOLD) {
    auto num_threads = omp_get_max_threads();
    int vec_per_thread = std::ceil((float)n / vecsize / num_threads);
#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += vec_per_thread * vecsize) {
      auto vec_start = i0;
      auto vec_end = std::min(i0 + vec_per_thread * vecsize, (int)n);
      if (is_sym_quant) {
        quantize_block_sym(in_ptr0, vec_start, vec_end, out_sym_ptr0);
      } else {
        quantize_block(in_ptr0, vec_start, vec_end, out_ptr0);
      }
    }
  } else {
    if (is_sym_quant) {
      quantize_block_sym(in_ptr0, 0, n, out_sym_ptr0);
    } else {
      quantize_block(in_ptr0, 0, n, out_ptr0);
    }
  }
  return out;
#else
  return at::quantize_per_tensor(t.to(c10::kFloat), scale, zp, c10::kQUInt8);
#endif
}


template <long ldb, int qw_type, bool sym_quant_w>
struct Dequantize<
    int8_t,
    ldb,
    /*N_GROUP_SIZE*/ 16,
    qw_type,
    sym_quant_w,
    /*use_g_idx*/ false> {
  template <int quant_a_mode> 
  static inline void call(
      uint8_t* qB,
      long K,
      long N,
      int8_t* zps,
      int8_t* B,
      int32_t* compensation) {
#if defined(CPU_CAPABILITY_AVX512_VNNI)
    auto pqB = GetVLAPtr<uint8_t>(qB, {ldb, 2}); // [K/4,N,4] packed in 4-bit
    auto pB = GetVLAPtr<int8_t>(B, {ldb, 4}); // [K/4,N,4]
    __m256i ones = _mm256_set1_epi8(1);
    for (int n = 0; n < N; n += 16) {
      __m256i vzps_low, vzps_high;
      if constexpr (!sym_quant_w) {
        auto [zps_low, zps_high] = load_zps_4vnni(&zps[n]);
        vzps_low = zps_low;
        vzps_high = zps_high;
      }
      __m256i vcompensate[2];
      if constexpr (is_asymmetric_quant_a(quant_a_mode)) {
          vcompensate[0] = _mm256_setzero_si256();
          vcompensate[1] = _mm256_setzero_si256();
      }
      // TODO(jgong5): unroll k?
      for (int k = 0; k < K / 4; k++) {
        // TODO(jgong5): consider optimize the instruction sequence below, e.g,
        // use avx512? load 64 (N:16, K:4) int4 values from qB
        __m256i vb_low, vb_high;
        if constexpr (!sym_quant_w) {
          auto [low, high] = load_uint4_as_int8(pqB[k][n]);
          vb_high = _mm256_sub_epi8(high, vzps_high);
          vb_low = _mm256_sub_epi8(low, vzps_low);
        } else if constexpr (qw_type == WOQ_DTYPE_INT4) {
          auto [low, high] = load_sint4_as_int8(pqB[k][n]);
          vb_low = low;
          vb_high = high;
        } else {
          auto [low, high] = load_nf4_as_int8(pqB[k][n]);
          vb_low = low;
          vb_high = high;
        }
        if constexpr (is_asymmetric_quant_a(quant_a_mode)) {
          vcompensate[0] = _mm256_dpbusd_epi32(vcompensate[0], ones, vb_low);
          vcompensate[1] = _mm256_dpbusd_epi32(vcompensate[1], ones, vb_high);
        }
        // store vb to B
        _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(pB[k][n]), vb_low);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i_u*>(pB[k][n + 8]), vb_high);
      }
      if constexpr (is_asymmetric_quant_a(quant_a_mode)) {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i_u*>(&compensation[n]), vcompensate[0]);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i_u*>(&compensation[n + 8]), vcompensate[1]);
      }
    }
#else
  TORCH_CHECK(false, "not implemented");
#endif
  }
};


template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
    const at::BFloat16* __restrict__ A, const at::quint4x2* __restrict__ B, at::BFloat16* __restrict__ C,
    const uint8_t* __restrict__ Bz, const at::BFloat16* __restrict__ Bs,
    const float* __restrict__ bias, int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t strideBz, int64_t strideBs) {

    static_assert(BLOCK_N % 32 == 0);
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 16 * 4;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vc_master[ROWS * COLS];

    __m256i mask = _mm256_set1_epi8(0xF);  // lower 4 bit
    // w and z are in [0,15], hence (w-z) is in [-15,15]
    // we will add 15 to it to shift it to [0,30] for lookup table indexing
    __m256i fifteen = _mm256_set1_epi8(15);
    __m512i bf16_lut = _mm512_set_epi16(0x0000, 0x4170, 0x4160, 0x4150, 0x4140, 0x4130, 0x4120, 0x4110,
                                        0x4100, 0x40E0, 0x40C0, 0x40A0, 0x4080, 0x4040, 0x4000, 0x3F80,
                                        0x0000,-0x4080,-0x4000,-0x3FC0,-0x3F80,-0x3F60,-0x3F40,-0x3F20,
                                       -0x3F00,-0x3EF0,-0x3EE0,-0x3ED0,-0x3EC0,-0x3EB0,-0x3EA0,-0x3E90);
    __m512 scales[COLS];
    __m256i zeros[COLS * 2];
    // repeat interleave
    __m256i idx1 = _mm256_set_epi8(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
                                   23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16);
    __m256i idx0 = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10,  9,  9,  8,  8,
                                    7,  7,  6,  6,  5,  5,  4,  4,  3,  3,  2,  2,  1,  1,  0,  0);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb; // ldb * 2 >> 1;
    const int64_t gs2 = group_size >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc_master[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc_master[i] = _mm512_set1_ps(0.f);
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    // x * ((w - zeros) * scales)
    // = (x * (w - zeros)) * scales

    auto pre_compute = [&](auto i, int64_t kgs) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      vc[i] = _mm512_set1_ps(0.f);  // reset accumulator

      // load zeros and scales
      if constexpr (row == 0 && col % 2 == 0) {
        // Bz layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=uint8
        __m256i tmp = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(Bz + kgs * strideBz + col * 16));
        // (w - (z - 15)) = (w - z + 15)
        tmp = _mm256_sub_epi8(tmp, fifteen);
        zeros[col]   = _mm256_permutexvar_epi8(idx0, tmp);
        zeros[col+1] = _mm256_permutexvar_epi8(idx1, tmp);

        // Bs layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=bf16
        __m512i tmp2 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(Bs + kgs * strideBs + col * 16));
        scales[col]   = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp2, 0));
        scales[col+1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp2, 1));
      }
    };
    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0 && col % 2 == 0) {
        __m256i vb_u4 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(B + k * ldb + col * 16));

        // deinterleave and lookup to BF16
        __m256i vb_i8_lo = vb_u4 & mask;
        __m256i vb_i8_hi = _mm256_srli_epi16(vb_u4, 4) & mask;
        vb_i8_lo = _mm256_sub_epi8(vb_i8_lo, zeros[col]);
        vb_i8_hi = _mm256_sub_epi8(vb_i8_hi, zeros[col+1]);
        vb[col]   = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_lo), bf16_lut);
        vb[col+1] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_hi), bf16_lut);

        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    auto post_compute = [&](auto i, int64_t kgs) {
      vc_master[i] = _mm512_fmadd_ps(vc[i], scales[i % COLS], vc_master[i]);
    };
    for (int64_t k = 0; k < K2; k += gs2) {
      Unroll<ROWS * COLS>{}(pre_compute, k / gs2);
      for (int64_t k_offset = 0; k_offset < gs2; ++k_offset) {
        Unroll<ROWS * COLS>{}(compute, k + k_offset);
      }
      Unroll<ROWS * COLS>{}(post_compute, k / gs2);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      if constexpr (col % 2 == 0) {
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>(C + row * ldc + col * 16),
            (__m512i)(_mm512_cvtne2ps_pbh(vc_master[i + 1], vc_master[i])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start, C + mb_start * ldc + nb_start, \
        Bz + nb_start, Bs + nb_start, has_bias ? bias + nb_start : nullptr,  \
        K, group_size, lda, ldb, ldc, strideBz, strideBs);

template <typename scalar_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const at::quint4x2* __restrict__ B,
      scalar_t* __restrict__ C,
      const uint8_t* __restrict__ Bz,
      const scalar_t* __restrict__ Bs,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      int64_t M,
      int64_t N,
      int64_t K,
      int group_size,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t strideBz,
      int64_t strideBs) {
    TORCH_CHECK(false, "struct brgemm: primary template not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)

// convert packed 8-bit integers to packed 32-bit integers
inline __m512 CVT_INT8_TO_FP32(__m128i x) {
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(x));
}


inline void unpack_B(
  at::Half* __restrict__ Btmp,
  const at::quint4x2* __restrict__ packed_B,
  const uint8_t* __restrict__ Bz,
  const at::Half* __restrict__ Bs,
  int64_t N,
  int64_t K,
  int group_size,
  int64_t ldb,
  int64_t ldb_tmp,
  int64_t strideBz,
  int64_t strideBs) {
    TORCH_CHECK(false, "int4 unpack does not support fp16 yet.");
  }
inline void unpack_B(
  at::BFloat16* __restrict__ Btmp,
  const at::quint4x2  * __restrict__ packed_B,
  const uint8_t* __restrict__ Bz,
  const at::BFloat16* __restrict__ Bs,
  int64_t N,
  int64_t K,
  int group_size,
  int64_t ldb,
  int64_t ldb_tmp,
  int64_t strideBz,
  int64_t strideBs) {
  const int64_t K2 = K >> 1;
  const int64_t gs2 = group_size >> 1;
  const int64_t ldb2 = ldb; // ldb * 2 >> 1;
  const int64_t ldb_tmp2 = ldb_tmp;
  float* btmp_ptr = reinterpret_cast<float *>(Btmp);

  __m256i mask = _mm256_set1_epi8(0xF);  // lower 4 bit
  __m256i zeros[2];
  __m512 scales[4];
  // repeat interleave
  __m256i z_idx1 = _mm256_set_epi8(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
                                   23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16);
  __m256i z_idx0 = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10,  9,  9,  8,  8,
                                    7,  7,  6,  6,  5,  5,  4,  4,  3,  3,  2,  2,  1,  1,  0,  0);
  __m512i s_idx1 = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);
  __m512i s_idx0 = _mm512_set_epi32( 7,  7,  6,  6,  5,  5,  4,  4,  3,  3,  2,  2, 1, 1, 0, 0);

  for (int n = 0; n < N; n += 32) {
    for (int k = 0; k < K2; ++k) {
      if (k % gs2 == 0) {
        const int kgs = k / gs2;

        // Bz layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=uint8
        __m256i tmp = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(Bz + kgs * strideBz + n));
        zeros[0] = _mm256_permutexvar_epi8(z_idx0, tmp);
        zeros[1] = _mm256_permutexvar_epi8(z_idx1, tmp);

        // Bs layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=bf16
        __m512i tmp2 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(Bs + kgs * strideBs + n));
        __m512 scales_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp2, 0));
        __m512 scales_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp2, 1));
        scales[0] = _mm512_permutexvar_ps(s_idx0, scales_lo);
        scales[1] = _mm512_permutexvar_ps(s_idx1, scales_lo);
        scales[2] = _mm512_permutexvar_ps(s_idx0, scales_hi);
        scales[3] = _mm512_permutexvar_ps(s_idx1, scales_hi);
      }

      __m256i vb_u4 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(packed_B + k * ldb2 + n));

      // deinterleave and subtract zero point
      __m256i vb_i8_lo = vb_u4 & mask;
      __m256i vb_i8_hi = _mm256_srli_epi16(vb_u4, 4) & mask;
      vb_i8_lo = _mm256_sub_epi8(vb_i8_lo, zeros[0]);
      vb_i8_hi = _mm256_sub_epi8(vb_i8_hi, zeros[1]);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
      // convert to FP32 and apply scales
      __m512 vb_f32_00 = CVT_INT8_TO_FP32(_mm256_extracti32x4_epi32(vb_i8_lo, 0)) * scales[0];
      __m512 vb_f32_01 = CVT_INT8_TO_FP32(_mm256_extracti32x4_epi32(vb_i8_lo, 1)) * scales[1];
      __m512 vb_f32_10 = CVT_INT8_TO_FP32(_mm256_extracti32x4_epi32(vb_i8_hi, 0)) * scales[2];
      __m512 vb_f32_11 = CVT_INT8_TO_FP32(_mm256_extracti32x4_epi32(vb_i8_hi, 1)) * scales[3];
#pragma GCC diagnostic pop

      __m512bh vb_bf16_0 = _mm512_cvtne2ps_pbh(vb_f32_01, vb_f32_00);
      __m512bh vb_bf16_1 = _mm512_cvtne2ps_pbh(vb_f32_11, vb_f32_10);
      _mm512_storeu_si512(btmp_ptr + k * ldb_tmp2 + n, (__m512i)vb_bf16_0);
      _mm512_storeu_si512(btmp_ptr + k * ldb_tmp2 + n + 16, (__m512i)vb_bf16_1);
    }
  }
}

template <bool has_bias>
struct brgemm<at::BFloat16, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::quint4x2* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const uint8_t* __restrict__ Bz,
      const at::BFloat16* __restrict__ Bs,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      int64_t M,
      int64_t N,
      int64_t K,
      int group_size,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t strideBz,
      int64_t strideBs) {
    constexpr int BLOCK_N = block_size_n();
    const int ldb_tmp = BLOCK_N;
    at::native::cpublas::brgemm(
      M, N, K, lda, ldb_tmp, BLOCK_N, false, A, Btmp, Ctmp);
    // copy from Ctmp to C
    for (int64_t m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};
#endif

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::quint4x2* __restrict__ B,
    scalar_t* __restrict__ C,
    const uint8_t* __restrict__ Bz,
    const scalar_t* __restrict__ Bs,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int group_size,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t strideBz,
    int64_t strideBs,
    bool brg) {

  if (brg) {
    brgemm<scalar_t, has_bias>::apply(
      A, B, C, Bz, Bs, Btmp, Ctmp, bias, M, N, K,
      group_size, lda, ldb, ldc, strideBz, strideBs);
    return;
  }

  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x14: LAUNCH_TINYGEMM_KERNEL_NN(1, 64); break;
        // mb_size = 2
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x24: LAUNCH_TINYGEMM_KERNEL_NN(2, 64); break;
        // mb_size = 3
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x34: LAUNCH_TINYGEMM_KERNEL_NN(3, 64); break;
        // mb_size = 4
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        case 0x44: LAUNCH_TINYGEMM_KERNEL_NN(4, 64); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void int4_w4a16_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const at::quint4x2* __restrict__ w,
    const uint8_t* __restrict__ w_zeros,
    const scalar_t* __restrict__ w_scales,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int group_size,
    int64_t mat1_strideM,
    int64_t out_strideM) {

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // TODO: find this threshold
  const bool use_brgemm = M > 4;
  scalar_t* Btmp_start = nullptr;
  if (use_brgemm) {
    at::Tensor Btmp_t = at::empty(
      {N, K}, c10::CppTypeToScalarType<scalar_t>::value);
   Btmp_start = Btmp_t.data_ptr<scalar_t>();
  at::parallel_for(0, NB, 0, [&](int64_t begin, int64_t end) {
    int64_t nb{0};
    data_index_init(begin,  nb, NB);
    for (int64_t i = begin; i < end; ++i) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(N - nb_start, BLOCK_N);
      auto Btmp = Btmp_start + nb_start*K;
      for (int64_t k = 0; k < K; k += BLOCK_K) {
        int64_t kb_size = std::min(static_cast<int64_t>(BLOCK_K), K - k);
        const int64_t kgs = k / group_size;
        auto strideBz = N;
        auto strideBs = N;
        auto ldb = nb_size;
        auto Bz = w_zeros + nb_start;
        auto Bs = w_scales + nb_start;
        auto B = w + nb_start * K / 2;
        unpack_B(Btmp + k*BLOCK_N, B + (k >> 1) * ldb, Bz + kgs * strideBz, Bs + kgs * strideBs,
                 nb_size, kb_size, group_size, ldb, BLOCK_N, strideBz, strideBs);

      }
      data_index_step( nb, NB);
    }
  });
}

  // l2 cache block for n
  int64_t cache_blocks_nb = get_cache_blocks<scalar_t>(BLOCK_N, K);
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t begin_mb, int64_t end_mb, int64_t begin_nb, int64_t end_nb) {
      // for brgemm, use float32 for accumulate
      alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
      for (int64_t nbb = begin_nb; nbb < end_nb; nbb += cache_blocks_nb) {
      for (int64_t mb = begin_mb; mb < end_mb; ++mb) {
      for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, end_nb); ++nb) {
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);
        tinygemm_kernel<scalar_t, has_bias>(
            /*   A  */ x + mb_start * mat1_strideM,
            /*   B  */ w + nb_start * K / 2,  // divide by 2 since w is u4 packed in u8
            /*   C  */ out + mb_start * out_strideM + nb_start,
            /*  Bz  */ w_zeros + nb_start,
            /*  Bs  */ w_scales + nb_start,
            /* Btmp */ use_brgemm ? Btmp_start + nb_start*K : nullptr,
            /* Ctmp */ Ctmp,
            /* bias */ bias + nb_start,
            /*   M  */ mb_size,
            /*   N  */ nb_size,
            /*   K  */ K,
            /*  gs  */ group_size,
            /* lda  */ mat1_strideM,
            /* ldb  */ nb_size,
            /* ldc  */ out_strideM,
            /* sBz  */ N,
            /* sBs  */ N,
            /* brg  */ use_brgemm);
      }}}
      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

} // anonymous namespace

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::quint4x2* __restrict__ B,
    scalar_t* __restrict__ C,
    const uint8_t* __restrict__ Bz,
    const scalar_t* __restrict__ Bs,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int group_size,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t strideBz,
    int64_t strideBs,
    bool brg) {
  tinygemm_kernel<scalar_t, false>(A, B, C, Bz, Bs, Btmp, Ctmp, nullptr, M, N, K,
                                   group_size, lda, ldb, ldc, strideBz, strideBs, brg);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE)   \
  template void tinygemm_kernel<TYPE>(        \
      const TYPE* __restrict__ A,             \
      const at::quint4x2* __restrict__ B,     \
      TYPE* __restrict__ C,                   \
      const uint8_t* __restrict__ Bz,         \
      const TYPE* __restrict__ Bs,            \
      TYPE* __restrict__ Btmp,                \
      float* __restrict__ Ctmp,               \
      int64_t M,                              \
      int64_t N,                              \
      int64_t K,                              \
      int group_size,                         \
      int64_t lda,                            \
      int64_t ldb,                            \
      int64_t ldc,                            \
      int64_t strideBz,                       \
      int64_t strideBs,                       \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

// mat1     : [M, K]
// mat2     : [N, K] (appear as [N, K/2] in u8)
// w_zeros  : [K/gs, N]
// w_scales : [K/gs, N]
// bias     : [N]
// out      : [M, N]
//
at::Tensor int4_w4a16_linear(
    at::Tensor& x,
    at::Tensor& w,
    at::Tensor& w_zeros,
    at::Tensor& w_scales,
    std::optional<at::Tensor>& bias) {
  RECORD_FUNCTION(
    "sgl-kernel::int4_w4a16_linear", std::vector<c10::IValue>({x, w, w_zeros, w_scales, bias}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(w_zeros);
  CHECK_INPUT(w_scales);

  int64_t M = x.size(0);
  int64_t N = w.size(0);
  int64_t K = x.size(1);
  int group_size = K / w_zeros.size(0);
  CHECK_EQ(w.size(1), K / 2);  // u4 packed as u8
  CHECK_DIM(2, x);
  CHECK_DIM(2, w);

  auto out = at::empty({M, N}, x.options());

  // strides
  int64_t x_strideM = x.stride(0);
  int64_t out_strideM = out.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "int4pack_linear_kernel_impl", [&] {
    int4_w4a16_linear_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        reinterpret_cast<const at::quint4x2*>(w.data_ptr<uint8_t>()),
        w_zeros.data_ptr<uint8_t>(),
        w_scales.data_ptr<scalar_t>(),
        bias_data,
        M,
        N,
        K,
        group_size,
        x_strideM,
        out_strideM);
  });

  return out;
}
