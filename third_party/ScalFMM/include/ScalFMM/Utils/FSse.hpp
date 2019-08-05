// See LICENCE file at project root
#ifndef FSSE_HPP
#define FSSE_HPP

#include "FGlobal.hpp"
#ifndef SCALFMM_USE_SSE
#error The SSE header is included while SCALFMM_USE_SSE is turned OFF
#endif

#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  //SSE2
#include <pmmintrin.h> //SSE3
#ifdef __SSSE3__
#include <tmmintrin.h>  //SSSE3
#endif
#ifdef __SSSE4_1__
#include <smmintrin.h> // SSE4
#endif


#ifndef _mm_set_pd1
// Looks like clang's emmintrin.h doesn't have this alternate name.
// But _mm_set1_pd is an equivalent to _mm_set_pd1.
#define _mm_set_pd1 _mm_set1_pd
#endif


#ifdef __SSEPE_INTEL_COMPILER

inline __m128d& operator+=(__m128d& v1, const __m128d& v2){
    return (v1 = _mm_add_pd(v1, v2));
}

inline __m128d& operator-=(__m128d& v1, const __m128d& v2){
    return (v1 = _mm_sub_pd(v1, v2));
}

inline __m128d& operator*=(__m128d& v1, const __m128d& v2){
    return (v1 = _mm_mul_pd(v1, v2));
}

inline __m128d& operator/=(__m128d& v1, const __m128d& v2){
    return (v1 = _mm_div_pd(v1, v2));
}

inline __m128d operator+(const __m128d& v1, const __m128d& v2){
    return _mm_add_pd(v1, v2);
}

inline __m128d operator-(const __m128d& v1, const __m128d& v2){
    return _mm_sub_pd(v1, v2);
}

inline __m128d operator*(const __m128d& v1, const __m128d& v2){
    return _mm_mul_pd(v1, v2);
}

inline __m128d operator/(const __m128d& v1, const __m128d& v2){
    return _mm_div_pd(v1, v2);
}

inline __m128& operator+=(__m128& v1, const __m128& v2){
    return (v1 = _mm_add_ps(v1, v2));
}

inline __m128& operator-=(__m128& v1, const __m128& v2){
    return (v1 = _mm_sub_ps(v1, v2));
}

inline __m128& operator*=(__m128& v1, const __m128& v2){
    return (v1 = _mm_mul_ps(v1, v2));
}

inline __m128& operator/=(__m128& v1, const __m128& v2){
    return (v1 = _mm_div_ps(v1, v2));
}

inline __m128 operator+(const __m128& v1, const __m128& v2){
    return _mm_add_ps(v1, v2);
}

inline __m128 operator-(const __m128& v1, const __m128& v2){
    return _mm_sub_ps(v1, v2);
}

inline __m128 operator*(const __m128& v1, const __m128& v2){
    return _mm_mul_ps(v1, v2);
}

inline __m128 operator/(const __m128& v1, const __m128& v2){
    return _mm_div_ps(v1, v2);
}

#endif

#endif // FSSE_HPP
