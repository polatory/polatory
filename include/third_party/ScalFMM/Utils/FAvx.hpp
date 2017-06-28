// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
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
