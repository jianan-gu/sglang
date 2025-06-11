#ifndef _TLA_VEC_H_
#define _TLA_VEC_H_

#include <immintrin.h>
// A class for forced loop unrolling at compile time
// These macro utils and the small gemm intrinsics kernels are implemented
// based on the initial code by pujiang.he@intel.com.
template <int i>
struct compile_time_for {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    compile_time_for<i - 1>::op(function, args...);
    function(std::integral_constant<int, i - 1>{}, args...);
  }
};
template <>
struct compile_time_for<1> {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    function(std::integral_constant<int, 0>{}, args...);
  }
};
template <>
struct compile_time_for<0> {
  // 0 loops, do nothing
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {}
};
template <typename VT>
struct VecOps {
  using ST = float;
  static constexpr int VLEN = 4;
  static VT loadu(const void* p) {
    TORCH_CHECK(false, "should not reach here");
  }
  static void storeu(void* p, VT v) {
    TORCH_CHECK(false, "should not reach here");
  }
  static VT set1(ST v) {
    TORCH_CHECK(false, "should not reach here");
  }
  static VT setzero() {
    TORCH_CHECK(false, "should not reach here");
  }
  static VT fmadd(VT a, VT b, VT c) {
    TORCH_CHECK(false, "should not reach here");
  }
  static VT set_0_to_15() {
    TORCH_CHECK(false, "should not reach here");
  }
  static VT set_neg_8_to_7() {
    TORCH_CHECK(false, "should not reach here");
  }
  static VT set_nf4_lut() {
    TORCH_CHECK(false, "should not reach here");
  }
  static VT mul() {
    TORCH_CHECK(false, "should not reach here");
  }
};

template <typename ST>
struct VecType {
  using type = void*;
};

#ifdef __AVX512F__
template <>
struct VecOps<__m512> {
  using ST = float;
  static constexpr int VLEN = sizeof(__m512) / sizeof(ST);
  static inline __m512 loadu(const void* p) {
    return _mm512_loadu_ps(p);
  }
  static inline void storeu(void* p, __m512 v) {
    _mm512_storeu_ps(p, v);
  }
  static inline __m512 set1(ST v) {
    return _mm512_set1_ps(v);
  }
  static inline __m512 setzero() {
    return _mm512_setzero_ps();
  }
  static inline __m512 fmadd(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
  }
  static inline __m512 mul(__m512 a, __m512 b) {
    return _mm512_mul_ps(a, b);
  }
  static inline __m512 set_0_to_15() {
    return _mm512_set_ps(
        15.0f,
        14.0f,
        13.0f,
        12.0f,
        11.0f,
        10.0f,
        9.0f,
        8.0f,
        7.0f,
        6.0f,
        5.0f,
        4.0f,
        3.0f,
        2.0f,
        1.0f,
        0.0f);
  }
  static inline __m512 set_neg_8_to_7() {
    return _mm512_set_ps(
        7.0f,
        6.0f,
        5.0f,
        4.0f,
        3.0f,
        2.0f,
        1.0f,
        0.0f,
        -1.0f,
        -2.0f,
        -3.0f,
        -4.0f,
        -5.0f,
        -6.0f,
        -7.0f,
        -8.0f);
  }
  static inline __m512 set_nf4_lut() {
    return _mm512_set_ps(
        1.0f,
        0.7229568362236023,
        0.5626170039176941,
        0.44070982933044434,
        0.33791524171829224,
        0.24611230194568634,
        0.16093020141124725,
        0.07958029955625534,
        0.0f,
        -0.09105003625154495,
        -0.18477343022823334,
        -0.28444138169288635,
        -0.39491748809814453,
        -0.5250730514526367,
        -0.6961928009986877,
        -1.0f);
  }
};

template <>
struct VecType<float> {
  using type = __m512;
};

#endif

#ifdef __AVX512FP16__
template <>
struct VecOps<__m512h> {
  using ST = _Float16;
  static constexpr int VLEN = sizeof(__m512h) / sizeof(ST);
  static inline __m512h loadu(const void* p) {
    return _mm512_loadu_ph(p);
  }
  static inline void storeu(void* p, __m512h v) {
    _mm512_storeu_ph(p, v);
  }
  static inline __m512h set1(ST v) {
    return _mm512_set1_ph(v);
  }
  static inline __m512h set1(float v) {
    return _mm512_set1_ph(ST(v));
  }
  static inline __m512h setzero() {
    return _mm512_setzero_ph();
  }
  static inline __m512h fmadd(__m512h a, __m512h b, __m512h c) {
    return _mm512_fmadd_ph(a, b, c);
  }
  static inline __m512h mul(__m512h a, __m512h b) {
    return _mm512_mul_ph(a, b);
  }
  static inline __m512h set_0_to_15() {
    return _mm512_set_ph(
        static_cast<_Float16>(15.0f),
        static_cast<_Float16>(14.0f),
        static_cast<_Float16>(13.0f),
        static_cast<_Float16>(12.0f),
        static_cast<_Float16>(11.0f),
        static_cast<_Float16>(10.0f),
        static_cast<_Float16>(9.0f),
        static_cast<_Float16>(8.0f),
        static_cast<_Float16>(7.0f),
        static_cast<_Float16>(6.0f),
        static_cast<_Float16>(5.0f),
        static_cast<_Float16>(4.0f),
        static_cast<_Float16>(3.0f),
        static_cast<_Float16>(2.0f),
        static_cast<_Float16>(1.0f),
        static_cast<_Float16>(0.0f),
        static_cast<_Float16>(15.0f),
        static_cast<_Float16>(14.0f),
        static_cast<_Float16>(13.0f),
        static_cast<_Float16>(12.0f),
        static_cast<_Float16>(11.0f),
        static_cast<_Float16>(10.0f),
        static_cast<_Float16>(9.0f),
        static_cast<_Float16>(8.0f),
        static_cast<_Float16>(7.0f),
        static_cast<_Float16>(6.0f),
        static_cast<_Float16>(5.0f),
        static_cast<_Float16>(4.0f),
        static_cast<_Float16>(3.0f),
        static_cast<_Float16>(2.0f),
        static_cast<_Float16>(1.0f),
        static_cast<_Float16>(0.0f));
  }
  static inline __m512h set_neg_8_to_7() {
    return _mm512_set_ph(
        static_cast<_Float16>(7.0f),
        static_cast<_Float16>(6.0f),
        static_cast<_Float16>(5.0f),
        static_cast<_Float16>(4.0f),
        static_cast<_Float16>(3.0f),
        static_cast<_Float16>(2.0f),
        static_cast<_Float16>(1.0f),
        static_cast<_Float16>(0.0f),
        static_cast<_Float16>(-1.0f),
        static_cast<_Float16>(-2.0f),
        static_cast<_Float16>(-3.0f),
        static_cast<_Float16>(-4.0f),
        static_cast<_Float16>(-5.0f),
        static_cast<_Float16>(-6.0f),
        static_cast<_Float16>(-7.0f),
        static_cast<_Float16>(-8.0f),
        static_cast<_Float16>(7.0f),
        static_cast<_Float16>(6.0f),
        static_cast<_Float16>(5.0f),
        static_cast<_Float16>(4.0f),
        static_cast<_Float16>(3.0f),
        static_cast<_Float16>(2.0f),
        static_cast<_Float16>(1.0f),
        static_cast<_Float16>(0.0f),
        static_cast<_Float16>(-1.0f),
        static_cast<_Float16>(-2.0f),
        static_cast<_Float16>(-3.0f),
        static_cast<_Float16>(-4.0f),
        static_cast<_Float16>(-5.0f),
        static_cast<_Float16>(-6.0f),
        static_cast<_Float16>(-7.0f),
        static_cast<_Float16>(-8.0f));
  }
  static inline __m512h set_nf4_lut() {
    return _mm512_set_ph(
        static_cast<_Float16>(1.0f),
        static_cast<_Float16>(0.7229568362236023f),
        static_cast<_Float16>(0.5626170039176941f),
        static_cast<_Float16>(0.44070982933044434f),
        static_cast<_Float16>(0.33791524171829224f),
        static_cast<_Float16>(0.24611230194568634f),
        static_cast<_Float16>(0.16093020141124725f),
        static_cast<_Float16>(0.07958029955625534f),
        static_cast<_Float16>(0.0f),
        static_cast<_Float16>(-0.09105003625154495f),
        static_cast<_Float16>(-0.18477343022823334f),
        static_cast<_Float16>(-0.28444138169288635f),
        static_cast<_Float16>(-0.39491748809814453f),
        static_cast<_Float16>(-0.5250730514526367f),
        static_cast<_Float16>(-0.6961928009986877f),
        static_cast<_Float16>(-1.0f),
        static_cast<_Float16>(1.0f),
        static_cast<_Float16>(0.7229568362236023f),
        static_cast<_Float16>(0.5626170039176941f),
        static_cast<_Float16>(0.44070982933044434f),
        static_cast<_Float16>(0.33791524171829224f),
        static_cast<_Float16>(0.24611230194568634f),
        static_cast<_Float16>(0.16093020141124725f),
        static_cast<_Float16>(0.07958029955625534f),
        static_cast<_Float16>(0.0f),
        static_cast<_Float16>(-0.09105003625154495f),
        static_cast<_Float16>(-0.18477343022823334f),
        static_cast<_Float16>(-0.28444138169288635f),
        static_cast<_Float16>(-0.39491748809814453f),
        static_cast<_Float16>(-0.5250730514526367f),
        static_cast<_Float16>(-0.6961928009986877f),
        static_cast<_Float16>(-1.0f));
  }
};

template <>
struct VecType<at::Half> {
  using type = __m512h;
};

template <>
struct VecType<_Float16> {
  using type = __m512h;
};

template <>
struct VecType<at::BFloat16> {
  using type = __m512;
};

#endif

#ifdef __AVX512F__
// load 32 bfloat16 values as 1 tuple of 2 float32 __m512 registers: (low, high)
inline std::tuple<__m512, __m512> _vec_load_bfloat16_as_two_floats(
    const at::BFloat16* addr) {
  auto cvt_bf16_to_fp32 = [](__m256i src) -> __m512 {
    auto y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
  };

  __m512i v = _mm512_loadu_si512(addr);
  // convert lower 16 bfloat16 values to float32
  auto v0 = cvt_bf16_to_fp32(_mm512_castsi512_si256(v));
  // convert higher 16 bfloat16 values to float32
  auto v1 = cvt_bf16_to_fp32(_mm512_extracti64x4_epi64(v, 1));
  return std::make_tuple(v0, v1);
};

// store 32 bfloat16 values from 2 float32 __m512 registers: v0: low, v1: high
inline void _vec_store_two_floats_as_bfloat16(
    at::BFloat16* addr,
    __m512 v0,
    __m512 v1) {
#ifdef __AVX512BF16__
  // convert lower 16 float32 values to bfloat16
  auto v0_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v0));
  // convert higher 16 float32 values to bfloat16
  auto v1_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v1));
  // combine the lower 16 and higher 16 bfloat16 values
  auto v = _mm512_castsi256_si512(v0_bf16);
  v = _mm512_inserti64x4(v, v1_bf16, 1);
  _mm512_storeu_si512(addr, v);
#else
  // TODO(jgong5): emuclate AVX512BF16 downcast
  TORCH_CHECK(false, "not implemented");
#endif
};

inline __m512i _vec_convert_two_floats_as_bfloat16(__m512 v0, __m512 v1) {
#ifdef __AVX512BF16__
  // convert lower 16 float32 values to bfloat16
  auto v0_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v0));
  // convert higher 16 float32 values to bfloat16
  auto v1_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v1));
  // combine the lower 16 and higher 16 bfloat16 values
  auto v = _mm512_castsi256_si512(v0_bf16);
  v = _mm512_inserti64x4(v, v1_bf16, 1);
  return v;
#else
  // TODO(jgong5): emuclate AVX512BF16 downcast
  TORCH_CHECK(false, "not implemented");
#endif
};
#endif // __AVX512F__

// TODO(jgong5): support prefetch hint?
template <long N, typename T>
struct VecArray {
  using vec_type = typename VecType<T>::type;
  using vec_ops = VecOps<vec_type>;
  static constexpr long n = N;
  static constexpr long num_vec = N / vec_ops::VLEN;
  using type = typename std::array<typename VecType<T>::type, num_vec>;

  static inline type load1d(T* p) {
    type result;
    compile_time_for<num_vec>::op(
        [&](auto i) { result[i] = vec_ops::loadu(p + i * vec_ops::VLEN); });
    return result;
  }
};

template <long N>
struct VecArray<N, at::BFloat16> {
  using vec_type = typename VecType<float>::type;
  using vec_ops = VecOps<vec_type>;
  static constexpr long num_vec = N / vec_ops::VLEN;
  using type = typename std::array<typename VecType<float>::type, num_vec>;

  static inline type load1d(at::BFloat16* p) {
    type result;
    compile_time_for<num_vec / 2>::op([&](auto i) {
      std::tie(result[i * 2], result[i * 2 + 1]) =
          _vec_load_bfloat16_as_two_floats(p + 32 * i);
    });
    return result;
  }
};

#endif