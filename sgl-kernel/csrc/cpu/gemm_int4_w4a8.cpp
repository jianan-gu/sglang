#include "common.h"
#include "vec.h"
#include "gemm.h"
#include "vec_struct.h"
namespace {

#define QUANT_A_THRESHOLD 30720
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))
constexpr long LOOP_K_UNROLL = 4; // TODO(jgong5): do not hard-code
constexpr int get_n_group_size(int N) {
  return N == 16 ? 16 : (N == 32 ? 32 : 64);
}
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))
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

template <long N_GROUP_SIZE, bool sym_quant>
struct load_dequant_zp_only_4bit {
  template <typename LUT, typename VAT>
  static inline VAT call(uint8_t* p, LUT lut, VAT vzps) {
    TORCH_CHECK(false, "not implemented");
  }
};

template <bool sym_quant>
struct load_dequant_zp_only_4bit<64, sym_quant> {
// TODO(jgong5): further simplify the dequant intrinsics below with VecOps
#if defined(CPU_CAPABILITY_AVX512)
  static inline std::array<__m512, 4> call(
      uint8_t* p,
      __m512 lut,
      std::array<__m512, 4> vzps) {
    using T = float;
    using VA = VecArray<64, T>;
    using VAT = typename VA::type;
    constexpr long COLS = VA::num_vec;
    auto packed = _mm256_loadu_si256((__m256i*)p);
    __m512i int32[COLS];
    {
      auto low_4bit = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
      auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
      int32[0] = low_4bit;
      int32[2] = high_4bit;
    }
    {
      auto low_4bit = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
      auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
      int32[1] = low_4bit;
      int32[3] = high_4bit;
    }
    VAT vbs;
    compile_time_for<COLS>::op([&](auto idx) {
      vbs[idx] = _mm512_permutexvar_ps(int32[idx], lut);
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ps(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif

#if defined(CPU_CAPABILITY_AVX512_FP16)
  static inline std::array<__m512h, 2> call(
      uint8_t* p,
      __m512h lut,
      std::array<__m512h, 2> vzps) {
    using T = tpp::half;
    using VA = VecArray<64, T>;
    using VAT = typename VA::type;
    constexpr long COLS = VA::num_vec;
    auto packed = _mm256_loadu_si256((__m256i*)p);
    __m512i int32[COLS];
    {
      auto low_4bit = _mm512_cvtepu8_epi16(packed);
      auto high_4bit = _mm512_srli_epi16(low_4bit, 4);
      int32[0] = low_4bit;
      int32[1] = high_4bit;
    }
    VAT vbs;
    compile_time_for<COLS>::op([&](auto idx) {
      vbs[idx] = _mm512_permutexvar_ph(int32[idx], lut);
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ph(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif
};

template <bool sym_quant>
struct load_dequant_zp_only_4bit<32, sym_quant> {
#if defined(CPU_CAPABILITY_AVX512)
  static inline std::array<__m512, 2> call(
      uint8_t* p,
      __m512 lut,
      std::array<__m512, 2> vzps) {
    using T = float;
    using VA = VecArray<32, T>;
    using VAT = typename VA::type;
    constexpr long COLS = VA::num_vec;
    auto packed = _mm_loadu_si128((__m128i*)p);
    __m512i int32[COLS];
    {
      auto low_4bit = _mm512_cvtepu8_epi32(packed);
      auto high_4bit = _mm512_srli_epi32(low_4bit, 4);
      int32[0] = low_4bit;
      int32[1] = high_4bit;
    }
    VAT vbs;
    compile_time_for<COLS>::op([&](auto idx) {
      vbs[idx] = _mm512_permutexvar_ps(int32[idx], lut);
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ps(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif

#if defined(CPU_CAPABILITY_AVX512_FP16)
  static inline std::array<__m512h, 1> call(
      uint8_t* p,
      __m512h lut,
      std::array<__m512h, 1> vzps) {
    using T = tpp::half;
    using VA = VecArray<32, T>;
    using VAT = typename VA::type;
    constexpr long COLS = VA::num_vec;
    auto packed = _mm_loadu_si128((__m128i*)p);
    __m512i int32[COLS];
    {
      auto low_4bit = _mm256_cvtepu8_epi16(packed);
      auto high_4bit = _mm256_srli_epi16(low_4bit, 4);
      // combine low_4bit and high_4bit into __m512i
      int32[0] =
          _mm512_inserti64x4(_mm512_castsi256_si512(low_4bit), high_4bit, 1);
    }
    VAT vbs;
    compile_time_for<COLS>::op([&](auto idx) {
      vbs[idx] = _mm512_permutexvar_ph(int32[idx], lut);
      if constexpr (!sym_quant) {
        vbs[idx] = _mm512_sub_ph(vbs[idx], vzps[idx]);
      }
    });
    return vbs;
  }
#endif
};

template <bool sym_quant>
struct load_dequant_zp_only_4bit<16, sym_quant> {
#if defined(CPU_CAPABILITY_AVX512)
  static inline std::array<__m512, 1> call(
      uint8_t* p,
      __m512 lut,
      std::array<__m512, 1> vzps) {
    using T = float;
    using VA = VecArray<16, T>;
    using VAT = typename VA::type;
    constexpr long COLS = VA::num_vec;
    static_assert(COLS == 1, "COLS must be 1");
    uint64_t packed = reinterpret_cast<uint64_t*>(p)[0];
    uint64_t high = packed >> 4;
    __m128i packed_128 = _mm_set_epi64x(high, packed);
    __m512i int32 = _mm512_cvtepu8_epi32(packed_128);
    VAT vbs;
    vbs[0] = _mm512_permutexvar_ps(int32, lut);
    if constexpr (!sym_quant) {
      vbs[0] = _mm512_sub_ps(vbs[0], vzps[0]);
    }
    return vbs;
  }
#endif

#if defined(CPU_CAPABILITY_AVX512_FP16)
  static inline std::array<__m512h, 0> call(
      uint8_t* p,
      __m512h lut,
      std::array<__m512h, 0> vzps) {
    TORCH_CHECK(false, "not implemented");
  }
#endif
};


template <long N, bool sym_quant, typename T>
struct load_dequant_4bit {
  using VT = typename VecType<T>::type;
  using V = VecOps<VT>;
  using VA = VecArray<N, T>;
  using VAT = typename VA::type;
  constexpr static long COLS = VA::num_vec;

  static inline VAT call(uint8_t* p, VAT vscales, VT lut, VAT vzps) {
    auto vbs = load_dequant_zp_only_4bit<N, sym_quant>::call(p, lut, vzps);
    compile_time_for<COLS>::op(
        [&](auto idx) { vbs[idx] = V::mul(vbs[idx], vscales[idx]); });
    return vbs;
  }
};

template <
    typename T,
    typename Tout,
    typename TScale,
    typename TZero,
    long M,
    long N,
    long ldb,
    bool transA = false,
    bool ACC = false,
    int quant_a_mode = -1,
    long PREFETCH_K_DIST = 0,
    typename Enabled = void>
struct GemmMicroKernel {
  template <int qw_type, bool sym_quant_w>
  static inline void call(
      long K,
      T* A,
      long lda,
      uint8_t* B,
      Tout* C,
      long ldc,
      TScale* scales,
      TZero* zps) {
    TORCH_CHECK(false, "Not implemented");
  }
};

template <
    typename T,
    long M,
    long N,
    long ldb,
    bool transA,
    bool ACC,
    int quant_a_mode,
    long PREFETCH_K_DIST>
struct GemmMicroKernel<
    T,
    T,
    T,
    T,
    M,
    N,
    ldb,
    transA,
    ACC,
    quant_a_mode,
    PREFETCH_K_DIST,
    typename std::enable_if_t<
        std::is_same<T, float>::value || std::is_same<T, at::Half>::value>> {
  // TODO(jgong5): generalize this with pre/post op handlers
  template <int qw_type, bool sym_quant_w>
  static inline void call(
      long K,
      T* A,
      long lda,
      uint8_t* B,
      T* C,
      long ldc,
      T* scales,
      T* zps) {
    static_assert(N % 16 == 0, "N must be a multiple of 16");
    constexpr const int N_GROUP_SIZE = get_n_group_size(N);

    using VT = typename VecType<T>::type;
    using V = VecOps<VT>;
    using ST = typename V::ST;
    using VArray = VecArray<N_GROUP_SIZE, T>;
    using VArrayT = typename VArray::type;

    constexpr const int COLS = N / V::VLEN;
    constexpr const int CBLOCK = N_GROUP_SIZE / V::VLEN;
    constexpr const int CNBLOCKS = N / N_GROUP_SIZE;
    VT va[M];
    VArrayT vb[CNBLOCKS];
    VT vc[M * COLS];
    VArrayT vscales[CNBLOCKS];
    VArrayT vzps[CNBLOCKS];

    VT lut;
    constexpr bool is_4bit_flag = true;
    if constexpr (is_4bit_flag) {
      if constexpr (sym_quant_w) {
        lut = V::set_neg_8_to_7();
      } else {
        lut = V::set_0_to_15();
      }
    }

    // Load scales and zps
    compile_time_for<CNBLOCKS>::op([&](auto i) {
      constexpr const int col = i * CBLOCK;
      vscales[i] = VArray::load1d(scales + col * V::VLEN);
      if constexpr (!sym_quant_w) {
        vzps[i] = VArray::load1d(zps + col * V::VLEN);
      }
    });

    // NB: For fp16 in int8 woq, we do not delay the scale to the post-op but
    // leave it to the dequant otherwise the weight value might be too large to
    // overflow fp16 range.
    constexpr bool scale_as_post_op = !std::is_same<T, at::Half>() || is_4bit_flag;

    compile_time_for<M * COLS>::op([&](auto i) { vc[i] = V::setzero(); });

    auto compute = [&](auto i, int k) {
      constexpr const int row = i / CNBLOCKS;
      constexpr const int cbidx = i % CNBLOCKS;

      if constexpr (cbidx == 0) {
        if constexpr (transA) {
          va[row] = V::set1(*(ST*)ADDRESS(A, k, row, lda));
        } else {
          va[row] = V::set1(*(ST*)ADDRESS(A, row, k, lda));
        }
      }

      if constexpr (row == 0) {
        constexpr const int col = cbidx * CBLOCK;
        if constexpr (scale_as_post_op) {
          if constexpr (is_4bit_flag) {
            vb[cbidx] =
                load_dequant_zp_only_4bit<N_GROUP_SIZE, sym_quant_w>::call(
                    ADDRESS(B, k, col * V::VLEN / 2, ldb / 2),
                    lut,
                    vzps[cbidx]);
          }
        } else {
          if constexpr (is_4bit_flag) {
            vb[cbidx] = load_dequant_4bit<N_GROUP_SIZE, sym_quant_w, T>::call(
                ADDRESS(B, k, col * V::VLEN / 2, ldb / 2),
                vscales[cbidx],
                lut,
                vzps[cbidx]);
          }
        }
        if constexpr (PREFETCH_K_DIST > 0) {
          if constexpr (is_4bit_flag) {
            _mm_prefetch(
                ADDRESS(B, k + PREFETCH_K_DIST, col * V::VLEN / 2, ldb / 2),
                _MM_HINT_T0);
          } else {
            _mm_prefetch(
                ADDRESS(B, k + PREFETCH_K_DIST, col * V::VLEN, ldb),
                _MM_HINT_T0);
          }
        }
      }

      compile_time_for<CBLOCK>::op([&](auto col) {
        constexpr const int idx = INDEX(row, INDEX(cbidx, col, CBLOCK), COLS);
        vc[idx] = V::fmadd(va[row], vb[cbidx][col], vc[idx]);
      });
    };

    // Accumulate along k
    constexpr const int unroll = LOOP_K_UNROLL;
    int k = 0;
    for (; k < K / unroll; k++) {
      compile_time_for<unroll>::op([&](auto i) {
        compile_time_for<M * CNBLOCKS>::op(compute, k * unroll + i);
      });
    }
    k *= unroll;
    for (; k < K; k++) {
      compile_time_for<M * CNBLOCKS>::op(compute, k);
    }

    // Store to C
    auto store = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      if constexpr (ACC) {
        auto vc_old = V::loadu(ADDRESS(C, row, col * V::VLEN, ldc));
        if constexpr (scale_as_post_op) {
          vc[i] = V::fmadd(vscales[col / CBLOCK][col % CBLOCK], vc[i], vc_old);
        } else {
          vc[i] = V::fmadd(V::set1(1.0f), vc[i], vc_old);
        }
      } else if constexpr (scale_as_post_op) {
        vc[i] = V::mul(vscales[col / CBLOCK][col % CBLOCK], vc[i]);
      }
      V::storeu(ADDRESS(C, row, col * V::VLEN, ldc), vc[i]);
    };

    compile_time_for<M * COLS>::op(store);
  }
};

#if defined(CPU_CAPABILITY_AVX512_VNNI)
template <
    long M,
    long N,
    long ldb,
    bool transA,
    bool ACC,
    int quant_a_mode,
    long PREFETCH_K_DIST>
struct GemmMicroKernel<
    /*Tin*/ uint8_t,
    /*Tout*/ float,
    /*TScale*/ float,
    /*TZero*/ int8_t,
    M,
    N,
    ldb,
    transA,
    ACC,
    quant_a_mode,
    PREFETCH_K_DIST> {
  template <int qw_type, bool sym_quant_w>
  static inline void call(
      long K,
      uint8_t* A,
      long lda,
      uint8_t* B,
      float* C,
      long ldc,
      float* scales,
      int8_t* zps,
      float* scale_a,
      int32_t* zp_a,
      int32_t k_groups) {
    if constexpr (!sym_quant_w) {
      TORCH_CHECK(zps, "Zero points must be given for asymmetric quantization");
    }
    auto pqB = GetVLAPtr<uint8_t>(B, {ldb, 2}); // [K/4,N,4] packed in 4-bit

    static_assert(N % 16 == 0, "N must be a multiple of 16");
    constexpr const int COLS = N / 16;

    __m512i ones = _mm512_set1_epi8(1); // used for computing compensation
    __m512i va;
    __m512i vb[COLS];
    __m512i vc[M * COLS];
    __m512 vscales[COLS];
    __m512i vzps[COLS];
    __m512i vcompensate[COLS];

    // Load scales and zps
    compile_time_for<COLS>::op([&](auto i) {
      vscales[i] = _mm512_loadu_ps(scales + i * 16);
      if constexpr (qw_type == WOQ_DTYPE_NF4) {
        const __m512 factor = _mm512_set1_ps(1.0f / 127.0f);
        vscales[i] = _mm512_mul_ps(vscales[i], factor);
      }
      // TODO(jgong5): should we use 512 or two 256 here?
      if constexpr (!sym_quant_w) {
        vzps[i] = combine_m256i(load_zps_4vnni(zps + i * 16));
      }
      if constexpr (is_asymmetric_quant_a(quant_a_mode)) {
        vcompensate[i] = _mm512_setzero_epi32();
      }
    });

    compile_time_for<M * COLS>::op(
        [&](auto i) { vc[i] = _mm512_setzero_epi32(); });

    auto compute = [&](auto i, int k) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;

      if constexpr (col == 0) {
        if constexpr (transA) {
          va = _mm512_set1_epi32(*(int32_t*)ADDRESS(A, k, row, lda));
        } else {
          va = _mm512_set1_epi32(*(int32_t*)ADDRESS(A, row, k, lda));
        }
      }

      if constexpr (row == 0) {
        if constexpr (!sym_quant_w) {
          vb[col] = combine_m256i(load_uint4_as_int8(pqB[k / 4][col * 16]));
          vb[col] = _mm512_sub_epi8(vb[col], vzps[col]);
        } else if constexpr (qw_type == WOQ_DTYPE_INT4) {
          vb[col] = combine_m256i(load_sint4_as_int8(pqB[k / 4][col * 16]));
        } else {
          vb[col] = combine_m256i(load_nf4_as_int8(pqB[k / 4][col * 16]));
        }
        if constexpr (is_asymmetric_quant_a(quant_a_mode)) {
          vcompensate[col] =
              _mm512_dpbusd_epi32(vcompensate[col], ones, vb[col]);
        }
        if constexpr (PREFETCH_K_DIST > 0) {
          _mm_prefetch(pqB[(k + PREFETCH_K_DIST) / 4][col * 16], _MM_HINT_T0);
        }
      }
      if constexpr (is_asymmetric_quant_a(quant_a_mode)) {
        vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
      } else {
        auto vsb = _mm512_sign_epi8(vb[col], va);
        auto vabsa = _mm512_sign_epi8(va, va);
        vc[i] = _mm512_dpbusds_epi32(vc[i], vabsa, vsb);
      }
    };

    // Accumulate along k
    constexpr const int unroll = LOOP_K_UNROLL;
    int k = 0;
    for (; k < K / 4 / unroll; k++) {
      compile_time_for<unroll>::op([&](auto i) {
        compile_time_for<M * COLS>::op(compute, 4 * (k * unroll + i));
      });
    }
    k *= 4 * unroll;
    for (; k < K; k += 4) {
      compile_time_for<M * COLS>::op(compute, k);
    }

    // Store to C
    auto store = [&](auto i) {
      constexpr const int row = i / COLS;
      constexpr const int col = i % COLS;
      // compute (qC - compensate * zp_a) * scale_a * scale_b
      // where compensate = sum(qB)
      __m512 vc_float;
      if constexpr (
          quant_a_mode == QUANT_A_PER_TENSOR ||
          quant_a_mode == QUANT_A_PER_K_BLOCK ||
          quant_a_mode == QUANT_A_PER_TENSOR_SYM ||
          quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
        if constexpr (
            quant_a_mode == QUANT_A_PER_TENSOR ||
            quant_a_mode == QUANT_A_PER_K_BLOCK) {
          vc[i] = _mm512_sub_epi32(
              vc[i],
              _mm512_mullo_epi32(vcompensate[col], _mm512_set1_epi32(*zp_a)));
        }
        vc_float = _mm512_cvtepi32_ps(vc[i]);
        vc_float = _mm512_mul_ps(vc_float, _mm512_set1_ps(*scale_a));
      } else if constexpr (
          quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
        if constexpr (quant_a_mode == QUANT_A_PER_M) {
          vc[i] = _mm512_sub_epi32(
              vc[i],
              _mm512_mullo_epi32(
                  vcompensate[col], _mm512_set1_epi32(*(zp_a + row))));
        }
        vc_float = _mm512_cvtepi32_ps(vc[i]);
        vc_float = _mm512_mul_ps(vc_float, _mm512_set1_ps(*(scale_a + row)));
      } else {
        if constexpr (is_asymmetric_quant_a(quant_a_mode)) {
          vc[i] = _mm512_sub_epi32(
              vc[i],
              _mm512_mullo_epi32(
                  vcompensate[col],
                  _mm512_set1_epi32(*(zp_a + row * k_groups))));
        }
        vc_float = _mm512_cvtepi32_ps(vc[i]);
        vc_float = _mm512_mul_ps(
            vc_float, _mm512_set1_ps(*(scale_a + row * k_groups)));
      }

      vc_float = _mm512_mul_ps(vc_float, vscales[col]);
      if constexpr (ACC) {
        auto vc_old = _mm512_loadu_ps(C + row * ldc + col * 16);
        vc_float = _mm512_add_ps(vc_float, vc_old);
      }
      _mm512_storeu_ps(C + row * ldc + col * 16, vc_float);
    };
    compile_time_for<M * COLS>::op(store);
  }
};
#endif

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
at::Tensor int4_w4w8_linear(
    at::Tensor& x,
    at::Tensor& w,
    at::Tensor& w_zeros,
    at::Tensor& w_scales,
    std::optional<at::Tensor>& bias) {
  RECORD_FUNCTION(
    "sgl-kernel::int4_w4w8_linear", std::vector<c10::IValue>({x, w, w_zeros, w_scales, bias}));

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
