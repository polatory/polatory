/*
 * FFortranMangling.hpp
 *
 *  Created on: 6 juin 2016
 *      Author: coulaud
 */

#ifndef SRC_UTILS_FFORTRANMANGLING_HPP_
#define SRC_UTILS_FFORTRANMANGLING_HPP_


#include "ScalFmmConfig.h"

#ifdef SCALFMM_BLAS_ADD_
/* Mangling for Fortran subroutine symbols with underscores. */

#define FortranName(name,NAME) name##_

#elif defined(SCALFMM_BLAS_UPCASE)

/* Mangling for Fortran subroutine  symbols in uppercase and without underscores */

#define FortranName(name,NAME) NAME

#elif defined(SCALFMM_BLAS_NOCHANGE)
/* Mangling for Fortran subroutine  symbols without no change. */

#define FortranName(name,NAME) name

#else

#error("Fortran MANGLING NOT DEFINED")

#endif

  // blas 1
#define Fddot  FortranName(ddot,DDOT)
#define Fdscal FortranName(dscal,DSCAL)
#define Fdcopy FortranName(dcopy,DCOPY)
#define Fdaxpy FortranName(daxpy,DAXPY)
#define Fsdot  FortranName(sdot,SDOT)
#define Fsscal FortranName(sscal,SSCAL)
#define Fscopy FortranName(scopy,SCOPY)
#define Fsaxpy FortranName(saxpy,SAXPY)
#define Fcscal FortranName(cscal,CSCAL)
#define Fccopy FortranName(ccopy,CCOPY)
#define Fcaxpy FortranName(caxpy,CAXPY)
#define Fzscal FortranName(zscal,ZSCAL)
#define Fzcopy FortranName(zcopy,ZCOPY)
#define Fzaxpy FortranName(zaxpy,ZAXPY)
// blas 2
#define Fdgemv FortranName(dgemv,DGEMV)
#define Fsgemv FortranName(sgemv,SGEMV)
#define Fcgemv FortranName(cgemv,CGEMV)
#define Fzgemv FortranName(zgemv,ZGEMV)
  // blas 3
#define Fdgemm FortranName(dgemm,DGEMM)
#define Fsgemm FortranName(sgemm,SGEMM)
#define Fcgemm FortranName(cgemm,CGEMM)
#define Fzgemm FortranName(zgemm,ZGEMM)
  // lapack
#define Fdgesvd FortranName(dgesvd,DGESVD)
#define Fdgeqrf FortranName(dgeqrf,DGEQRF)
#define Fdgeqp3  FortranName(dgeqp3,DGEQP3)
#define Fdorgqr  FortranName(dorgqr,DORGQR)
#define Fdormqr FortranName(dormqr,DORMQR)
#define Fdpotrf  FortranName(dpotrf,DPOTRF)
#define Fsgesvd FortranName(sgesvd,SGESVD)
#define Fsgeqrf FortranName(sgeqrf,SGEQRF)
#define Fsgeqp3  FortranName(sgeqp3,SGEQP3)
#define Fsorgqr  FortranName(sorgqr,SORGQR)
#define Fsormqr FortranName(sormqr,SORMQR)
#define Fspotrf  FortranName(spotrf,SPOTRF)
#define Fcgesvd FortranName(cgesvd,CGESVD)
#define Fcgeqrf FortranName(cgeqrf,CGEQRF)
#define Fcgeqp3  FortranName(cgeqp3,CGEQP3)
#define Fcorgqr  FortranName(corgqr,CORGQR)
#define Fcormqr FortranName(cormqr,CORMQR)
#define Fcpotrf  FortranName(cpotrf,CPOTRF)
#define Fzgesvd FortranName(zgesvd,ZGESVD)
#define Fzgeqrf FortranName(zgeqrf,ZGEQRF)
#define Fzgeqp3  FortranName(zgeqp3,ZGEQP3)
#define Fzorgqr  FortranName(zorgqr,ZORGQR)
#define Fzormqr FortranName(zormqr,ZORMQR)
#define Fzpotrf  FortranName(zpotrf,ZPOTRF)


#endif /* SRC_UTILS_FFORTRANMANGLING_HPP_ */



