// See LICENCE file at project root
#ifndef FAVX_HPP
#define FAVX_HPP

#include "FGlobal.hpp"
#ifndef SCALFMM_USE_AVX
#error The AVX header is included while SCALFMM_USE_AVX is turned OFF
#endif

#include "immintrin.h"

#ifdef __AVXPE_INTEL_COMPILER

//Side effect operators DOUBLE
inline __m256d& operator+=(__m256d & a, const __m256d & b){
  return (a = _mm256_add_pd (a,b));
}

inline __m256d& operator-=(__m256d& a, const __m256d& b){
  return (a = _mm256_sub_pd (a,b));
}

inline __m256d& operator*=(__m256d& a, const __m256d& b){
  return (a = _mm256_mul_pd (a,b));
}

inline __m256d& operator/=(__m256d& a, const __m256d& b){
  return (a = _mm256_div_pd (a,b));
}

//No side effect operators DOUBLE
inline __m256d operator+(const __m256d& a,const  __m256d& b){
  return _mm256_add_pd (a,b);
}

inline __m256d operator-(const __m256d& a, const __m256d& b){
  return _mm256_sub_pd (a,b);
}

inline __m256d operator*(const __m256d& v1, const __m256d& v2){
    return _mm256_mul_pd(v1, v2);
}

inline __m256d operator/(const __m256d& v1, const __m256d& v2){
    return _mm256_div_pd(v1, v2);
}

//Side effect operators SINGLE
inline __m256& operator+=(__m256 & a, const __m256 & b){
  return (a = _mm256_add_ps (a,b));
}

inline __m256& operator-=(__m256& a, const __m256& b){
  return (a = _mm256_sub_ps (a,b));
}

inline __m256& operator*=(__m256& a, const __m256& b){
  return (a = _mm256_mul_ps (a,b));
}

inline __m256& operator/=(__m256& a, const __m256& b){
  return (a = _mm256_div_ps (a,b));
}

//No side effect operators SINGLE
inline __m256 operator+(const __m256& a,const  __m256& b){
  return _mm256_add_ps (a,b);
}

inline __m256 operator-(const __m256& a, const __m256& b){
  return _mm256_sub_ps (a,b);
}

inline __m256 operator*(const __m256& v1, const __m256& v2){
    return _mm256_mul_ps(v1, v2);
}

inline __m256 operator/(const __m256& v1, const __m256& v2){
    return _mm256_div_ps(v1, v2);
}

#endif

#endif
