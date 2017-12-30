// See LICENCE file at project root
#ifndef FGLOBAL_HPP
#define FGLOBAL_HPP

#include "../ScalFmmConfig.h"

///////////////////////////////////////////////////////
// Memory profiling
///////////////////////////////////////////////////////

#include "FMemStats.h"

///////////////////////////////////////////////////////
// Stdlib
///////////////////////////////////////////////////////

#include <cstdlib>

///////////////////////////////////////////////////////
// Operating System
///////////////////////////////////////////////////////

#if defined(_WIN32) || defined(ming)
    #define WINDOWS
#else
    #define POSIX
#endif


///////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////

typedef long long int FSize;

///////////////////////////////////////////////////////
// Restrict
///////////////////////////////////////////////////////

static const int MaxTreeHeight = 20;


///////////////////////////////////////////////////////
// Morton index
///////////////////////////////////////////////////////

typedef long long MortonIndex;


///////////////////////////////////////////////////////
// Restrict
///////////////////////////////////////////////////////

#ifdef WINDOWS
    #define FRestrict __restrict
#else
    #define FRestrict __restrict__
#endif

///////////////////////////////////////////////////////
// Prefetch
///////////////////////////////////////////////////////

#ifdef SCALFMM_USE_SSE
    #ifdef __GNUC__
            #include <xmmintrin.h>
            #define Prefetch_Read0(X)  _mm_prefetch((char*)(X), _MM_HINT_T0);
            inline void Prefetch_Write0_core(const char* ptr){
                asm("prefetchw (%0)": : "g"(ptr) :);
            }
            #define Prefetch_Write0(X) Prefetch_Write0_core((const char*)X);
            #define Prefetch_Read1(X)  _mm_prefetch((char*)(X), _MM_HINT_T1);
            #define Prefetch_Write1(X) _mm_prefetch((char*)(X), _MM_HINT_T1);
            #define Prefetch_Read2(X)  _mm_prefetch((char*)(X), _MM_HINT_T2);
            #define Prefetch_Write2(X) _mm_prefetch((char*)(X), _MM_HINT_T2);
    #else
        #include <xmmintrin.h>
        #define Prefetch_Read0(X)  _mm_prefetch((char*)(X), _MM_HINT_T0);
        #define Prefetch_Write0(X) _mm_prefetch((char*)(X), _MM_HINT_T0);
        #define Prefetch_Read1(X)  _mm_prefetch((char*)(X), _MM_HINT_T1);
        #define Prefetch_Write1(X) _mm_prefetch((char*)(X), _MM_HINT_T1);
        #define Prefetch_Read2(X)  _mm_prefetch((char*)(X), _MM_HINT_T2);
        #define Prefetch_Write2(X) _mm_prefetch((char*)(X), _MM_HINT_T2);
    #endif
#else
    #define Prefetch_Read0(X)
    #define Prefetch_Write0(X)
    #define Prefetch_Read1(X)
    #define Prefetch_Write1(X)
    #define Prefetch_Read2(X)
    #define Prefetch_Write2(X)
#endif


///////////////////////////////////////////////////////
// Test OMP4
///////////////////////////////////////////////////////

#if _OPENMP >= 201307 && !defined(SCALFMM_DISABLE_NATIVE_OMP4)
#ifndef __INTEL_COMPILER
#define SCALFMM_USE_OMP4
#endif
#endif


///////////////////////////////////////////////////////
// Default P2P Alignement
///////////////////////////////////////////////////////

static const int FP2PDefaultAlignement = 64;


#endif //FGLOBAL_HPP

