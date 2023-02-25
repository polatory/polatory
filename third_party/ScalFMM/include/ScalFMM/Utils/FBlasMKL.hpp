// See LICENCE file at project root
#ifndef FBLASMKL_HPP
#define FBLASMKL_HPP

#include "mkl_blas.h"
#include "mkl_lapack.h"

#define as_c(x) reinterpret_cast<MKL_Complex8 *>(x)
#define as_z(x) reinterpret_cast<MKL_Complex16 *>(x)
#define as_const_c(x) reinterpret_cast<const MKL_Complex8 *>(x)
#define as_const_z(x) reinterpret_cast<const MKL_Complex16 *>(x)

#endif  // FBLASMKL_HPP
