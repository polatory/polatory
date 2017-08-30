#ifndef FCUDAGLOBAL_HPP
#define FCUDAGLOBAL_HPP

#include "../../Utils/FGlobal.hpp"

// Manage special case for nvcc
#if defined(__CUDACC__) || defined(__NVCC__)
#else
#endif

#include <cuda.h>

#include <cstdio>

static void FCudaCheckCore(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cudaGetErrorString(code), file, line);
      exit(code);
   }
}
#define FCudaCheck( test ) { FCudaCheckCore((test), __FILE__, __LINE__); }
#define FCudaCheckAfterCall() { FCudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }
#define FCudaAssertLF(ARGS) if(!(ARGS)){\
                                printf("Error line %d\n", __LINE__);\
                            }

#endif // FCUDAGLOBAL_HPP

