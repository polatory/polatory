// See LICENCE file at project root
#ifndef FBLAS_HPP
#define FBLAS_HPP

#include "mkl_blas.h"
#include "mkl_lapack.h"

#define as_c(x) reinterpret_cast<MKL_Complex8 *>(x)
#define as_z(x) reinterpret_cast<MKL_Complex16 *>(x)
#define as_const_c(x) reinterpret_cast<const MKL_Complex8 *>(x)
#define as_const_z(x) reinterpret_cast<const MKL_Complex16 *>(x)

// This file interfaces the blas functions
// to enable a generic use.
// If no blas has been enabled in the cmake,
// the function will be empty


// for real
namespace scalfmm {
  const double D_ZERO = 0.0;
  const double D_ONE  = 1.0;
  const double D_MONE = -1.0;
  const float  S_ZERO = 0.0;
  const float  S_ONE  = 1.0;
  const float  S_MONE = -1.0;
  // for complex
  const MKL_Complex16 Z_ZERO = MKL_Complex16{ 0.0, 0.0 };
  const MKL_Complex16 Z_ONE  = MKL_Complex16{ 1.0, 0.0 };
  const MKL_Complex16 Z_MONE = MKL_Complex16{ -1.0, 0.0 };
  const MKL_Complex8  C_ZERO = MKL_Complex8{ 0.0, 0.0 };
  const MKL_Complex8  C_ONE  = MKL_Complex8{ 1.0, 0.0 };
  const MKL_Complex8  C_MONE = MKL_Complex8{ -1.0, 0.0 };

  const int N_ONE = 1;
  const int N_MONE = -1;
  const char JOB_STR[] = "NTOSVULCR";
}


namespace FBlas {

  // copy
  inline void copy(int n, double *orig, double *dest)
  { dcopy(&n, orig, &scalfmm::N_ONE, dest, &scalfmm::N_ONE); }
  inline void copy(int n, float *orig, float *dest)
  { scopy(&n, orig, &scalfmm::N_ONE, dest, &scalfmm::N_ONE); }
  inline void c_copy(int n, double *orig, double *dest)
  { zcopy(&n, as_z(orig), &scalfmm::N_ONE, as_z(dest), &scalfmm::N_ONE); }
  inline void c_copy(int n, float *orig, float *dest)
  { ccopy(&n, as_c(orig), &scalfmm::N_ONE, as_c(dest), &scalfmm::N_ONE); }

  // copy (variable increment)
  inline void copy(int n, double *orig, int inco, double *dest, int incd)
  { dcopy(&n, orig, &inco, dest, &incd); }
  inline void copy(int n, float *orig, int inco, float *dest, int incd)
  { scopy(&n, orig, &inco, dest, &incd); }
  inline void c_copy(int n, double *orig, int inco, double *dest, int incd)
  { zcopy(&n, as_z(orig), &inco, as_z(dest), &incd); }
  inline void c_copy(int n, float *orig, int inco, float *dest, int incd)
  { ccopy(&n, as_c(orig), &inco, as_c(dest), &incd); }

  // scale
  inline void scal(int n, double d, double *x)
  { dscal(&n, &d, x, &scalfmm::N_ONE); }
  inline void scal(int n, float d, float *x)
  { sscal(&n, &d, x, &scalfmm::N_ONE); }
  inline void c_scal(int n, double d, double *x)
  { zscal(&n, as_z(&d), as_z(x), &scalfmm::N_ONE); }
  inline void c_scal(int n, float d, float *x)
  { cscal(&n, as_c(&d), as_c(x), &scalfmm::N_ONE); }

  // scale (variable increment)
  inline void scal(int n, double d, double *x, int incd)
  { dscal(&n, &d, x, &incd); }
  inline void scal(int n, float d, float *x, int incd)
  { sscal(&n, &d, x, &incd); }
  inline void c_scal(int n, double d, double *x, int incd)
  { zscal(&n, as_z(&d), as_z(x), &incd); }
  inline void c_scal(int n, float d, float *x, int incd)
  { cscal(&n, as_c(&d), as_c(x), &incd); }

  // set zero
  inline void setzero(const unsigned n, double *x)
  { for (unsigned i=0; i<n; ++i) x[i] = 0.0; }
  inline void setzero(const unsigned n, float *x)
  { for (unsigned i=0; i<n; ++i) x[i] = 0.0f; }
  inline void c_setzero(const unsigned n, double *x)
  { for (unsigned i=0; i<n; ++i) x[i*2] = x[i*2+1] = 0.0; }
  inline void c_setzero(const unsigned n, float *x)
  { for (unsigned i=0; i<n; ++i) x[i*2] = x[i*2+1] = 0.0f; }

  // y += x
  inline void add(const int n, double *x, double *y)
  { daxpy(&n, &scalfmm::D_ONE, x, &scalfmm::N_ONE, y, &scalfmm::N_ONE); }
  inline void add(const int n, float *x, float *y)
  { saxpy(&n, &scalfmm::S_ONE, x, &scalfmm::N_ONE, y, &scalfmm::N_ONE); }
  inline void c_add(const int n, float *x, float *y)
  { caxpy(&n, &scalfmm::C_ONE, as_c(x), &scalfmm::N_ONE, as_c(y), &scalfmm::N_ONE); }
  inline void c_add(const int n, double *x,double *y)
  { zaxpy(&n, &scalfmm::Z_ONE, as_z(x), &scalfmm::N_ONE, as_z(y), &scalfmm::N_ONE); }

  // y += d x
  inline void axpy(const int n, const double d, const double *x, double *y)
  { daxpy(&n, &d, x, &scalfmm::N_ONE, y, &scalfmm::N_ONE); }
  inline void axpy(const int n, const float d, const float *x, float *y)
  { saxpy(&n, &d, x, &scalfmm::N_ONE, y, &scalfmm::N_ONE); }
  inline void c_axpy(const int n, const float *d, const float *x, float *y)
  { caxpy(&n, as_const_c(d), as_const_c(x), &scalfmm::N_ONE, as_c(y), &scalfmm::N_ONE); }
  inline void c_axpy(const int n, const double *d, const double *x, double *y)
  { zaxpy(&n, as_const_z(d), as_const_z(x), &scalfmm::N_ONE, as_z(y), &scalfmm::N_ONE); }

  // y = d Ax
  inline void gemv(const int m, const int n, double d, double *A, double *x, double *y)
  { dgemv(scalfmm::JOB_STR, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::D_ZERO, y, &scalfmm::N_ONE); }
  inline void gemv(const int m, const int n, float d, float *A, float *x, float *y)
  { sgemv(scalfmm::JOB_STR, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::S_ZERO, y, &scalfmm::N_ONE); }
  inline void c_gemv(const int m, const int n, float *d, float *A, float *x, float *y)
  { cgemv(scalfmm::JOB_STR, &m, &n, as_c(d), as_c(A), &m, as_c(x), &scalfmm::N_ONE, &scalfmm::C_ZERO, as_c(y), &scalfmm::N_ONE); }
  inline void c_gemv(const int m, const int n, double *d, double *A, double *x, double *y)
  { zgemv(scalfmm::JOB_STR, &m, &n, as_z(d), as_z(A), &m, as_z(x), &scalfmm::N_ONE, &scalfmm::Z_ZERO, as_z(y), &scalfmm::N_ONE); }

  // y += d Ax
  inline void gemva(const int m, const int n, double d, double *A, double *x, double *y)
  { dgemv(scalfmm::JOB_STR, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::D_ONE, y, &scalfmm::N_ONE); }
  inline void gemva(const int m, const int n, float d, float *A, float *x, float *y)
  { sgemv(scalfmm::JOB_STR, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::S_ONE, y, &scalfmm::N_ONE); }
  inline void c_gemva(const int m, const int n, const float *d, const float *A, const float *x, float *y)
  { cgemv(scalfmm::JOB_STR, &m, &n, as_const_c(d), as_const_c(A), &m, as_const_c(x), &scalfmm::N_ONE, &scalfmm::C_ONE, as_c(y), &scalfmm::N_ONE); }
  inline void c_gemva(const int m, const int n, const double *d, const double *A, const double *x, double *y)
  { zgemv(scalfmm::JOB_STR, &m, &n, as_const_z(d), as_const_z(A), &m, as_const_z(x), &scalfmm::N_ONE, &scalfmm::Z_ONE, as_z(y), &scalfmm::N_ONE); }

  // y = d A^T x
  inline void gemtv(const int m, const int n, double d, double *A, double *x, double *y)
  { dgemv(scalfmm::JOB_STR+1, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::D_ZERO, y, &scalfmm::N_ONE); }
  inline void gemtv(const int m, const int n, float d, float *A, float *x, float *y)
  { sgemv(scalfmm::JOB_STR+1, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::S_ZERO, y, &scalfmm::N_ONE); }
  inline void c_gemtv(const int m, const int n, float *d, float *A, float *x, float *y)
  { cgemv(scalfmm::JOB_STR+1, &m, &n, as_c(d), as_c(A), &m, as_c(x), &scalfmm::N_ONE, &scalfmm::C_ZERO, as_c(y), &scalfmm::N_ONE); }
  inline void c_gemtv(const int m, const int n, double *d, double *A, double *x, double *y)
  { zgemv(scalfmm::JOB_STR+1, &m, &n, as_z(d), as_z(A), &m, as_z(x), &scalfmm::N_ONE, &scalfmm::Z_ZERO, as_z(y), &scalfmm::N_ONE); }
  inline void c_gemhv(const int m, const int n, float *d, float *A, float *x, float *y)
  { cgemv(scalfmm::JOB_STR+7, &m, &n, as_c(d), as_c(A), &m, as_c(x), &scalfmm::N_ONE, &scalfmm::C_ZERO, as_c(y), &scalfmm::N_ONE); } // hermitian transposed
  inline void c_gemhv(const int m, const int n, double *d, double *A, double *x, double *y)
  { zgemv(scalfmm::JOB_STR+7, &m, &n, as_z(d), as_z(A), &m, as_z(x), &scalfmm::N_ONE, &scalfmm::Z_ZERO, as_z(y), &scalfmm::N_ONE); } // hermitian transposed

  // y += d A^T x
  inline void gemtva(const int m, const int n, double d, double *A, double *x, double *y)
  { dgemv(scalfmm::JOB_STR+1, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::D_ONE, y, &scalfmm::N_ONE); }
  inline void gemtva(const int m, const int n, float d, float *A, float *x, float *y)
  { sgemv(scalfmm::JOB_STR+1, &m, &n, &d, A, &m, x, &scalfmm::N_ONE, &scalfmm::S_ONE, y, &scalfmm::N_ONE); }
  inline void c_gemtva(const int m, const int n, float *d, float *A, float *x, float *y)
  { cgemv(scalfmm::JOB_STR+1, &m, &n, as_c(d), as_c(A), &m, as_c(x), &scalfmm::N_ONE, &scalfmm::C_ONE, as_c(y), &scalfmm::N_ONE); }
  inline void c_gemtva(const int m, const int n, double *d, double *A, double *x, double *y)
  { zgemv(scalfmm::JOB_STR+1, &m, &n, as_z(d), as_z(A), &m, as_z(x), &scalfmm::N_ONE, &scalfmm::Z_ONE, as_z(y), &scalfmm::N_ONE); }
  inline void c_gemhva(const int m, const int n, float *d, float *A, float *x, float *y)
  { cgemv(scalfmm::JOB_STR+7, &m, &n, as_c(d), as_c(A), &m, as_c(x), &scalfmm::N_ONE, &scalfmm::C_ONE, as_c(y), &scalfmm::N_ONE); } // hermitian transposed
  inline void c_gemhva(const int m, const int n, double *d, double *A, double *x, double *y)
  { zgemv(scalfmm::JOB_STR+7, &m, &n, as_z(d), as_z(A), &m, as_z(x), &scalfmm::N_ONE, &scalfmm::Z_ONE, as_z(y), &scalfmm::N_ONE); } // hermitian transposed




  // C = d A B, A is m x p, B is p x n
  inline void gemm(int m, int p, int n, double d,
                   double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { dgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::D_ZERO, C, &ldC); }
  inline void gemm(int m, int p, int n, float d,
                   float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { sgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::S_ZERO, C, &ldC); }
  inline void c_gemm(const int m, const int p, const int n, const float *d,
                     float *A, const int ldA, float *B, const int ldB, float *C, const int ldC)
  { cgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, as_const_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ZERO, as_c(C), &ldC); }
  inline void c_gemm(const int m, const int p, const int n, const double *d,
                     double *A, const int ldA, double *B, const int ldB, double *C, const int ldC)
  { zgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, as_const_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ZERO, as_z(C), &ldC); }

  // C += d A B, A is m x p, B is p x n
  inline void gemma(int m, int p, int n, double d,
                    double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { dgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::D_ONE, C, &ldC); }
  inline void gemma(int m, int p, int n, float d,
                    float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { sgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::S_ONE, C, &ldC); }
  inline void c_gemma(int m, int p, int n, float *d,
                      float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, as_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ONE, as_c(C), &ldC); }
  inline void c_gemma(int m, int p, int n, double *d,
                      double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR, scalfmm::JOB_STR, &m, &n, &p, as_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ONE, as_z(C), &ldC); }

  // C = d A^T B, A is m x p, B is m x n
  inline void gemtm(int m, int p, int n, double d,
                    double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { dgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, &d, A, &ldA, B, &ldB, &scalfmm::D_ZERO, C, &ldC); }
  inline void gemtm(int m, int p, int n, float d,
                    float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { sgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, &d, A, &ldA, B, &ldB, &scalfmm::S_ZERO, C, &ldC); }
  inline void c_gemtm(int m, int p, int n, float *d,
                      float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, as_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ZERO, as_c(C), &ldC); }
  inline void c_gemtm(int m, int p, int n, double *d,
                      double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, as_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ZERO, as_z(C), &ldC); }
  inline void c_gemhm(int m, int p, int n, float *d, // hermitialn transposed
                      float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR+7, scalfmm::JOB_STR, &p, &n, &m, as_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ZERO, as_c(C), &ldC); }
  inline void c_gemhm(int m, int p, int n, double *d, // hermitian transposed
                      double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR+7, scalfmm::JOB_STR, &p, &n, &m, as_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ZERO, as_z(C), &ldC); }

  // C += d A^T B, A is m x p, B is m x n
  inline void gemtma(int m, int p, int n, double d,
                     double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { dgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, &d, A, &ldA, B, &ldB, &scalfmm::D_ONE, C, &ldC); }
  inline void gemtma(int m, int p, int n, float d,
                     float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { sgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, &d, A, &ldA, B, &ldB, &scalfmm::S_ONE, C, &ldC); }
  inline void c_gemtma(int m, int p, int n, float *d,
                       float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, as_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ONE, as_c(C), &ldC); }
  inline void c_gemtma(int m, int p, int n, double *d,
                       double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR+1, scalfmm::JOB_STR, &p, &n, &m, as_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ONE, as_z(C), &ldC); }
  inline void c_gemhma(int m, int p, int n, float *d, // hermitian transposed
                       float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR+7, scalfmm::JOB_STR, &p, &n, &m, as_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ONE, as_c(C), &ldC); }
  inline void c_gemhma(int m, int p, int n, double *d, // hermitian transposed
                       double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR+7, scalfmm::JOB_STR, &p, &n, &m, as_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ONE, as_z(C), &ldC); }


  // C = d A B^T, A is m x p, B is n x p
  inline void gemmt(int m, int p, int n, double d,
                    double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { dgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::D_ZERO, C, &ldC); }
  inline void gemmt(int m, int p, int n, float d,
                    float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { sgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::S_ZERO, C, &ldC); }
  inline void c_gemmt(int m, int p, int n, float d,
                      float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, as_c(&d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ZERO, as_c(C), &ldC); }
  inline void c_gemmt(int m, int p, int n, double d,
                      double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, as_z(&d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ZERO, as_z(C), &ldC); }
  inline void c_gemmh(int m, int p, int n, float d, // hermitian transposed
                      float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+7, &m, &n, &p, as_c(&d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ZERO, as_c(C), &ldC); }
  inline void c_gemmh(int m, int p, int n, double d, // hermitian transposed
                      double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+7, &m, &n, &p, as_z(&d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ZERO, as_z(C), &ldC); }

  // C += d A B^T, A is m x p, B is n x p
  inline void gemmta(int m, int p, int n, double d,
                     double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { dgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::D_ONE, C, &ldC); }
  inline void gemmta(int m, int p, int n, float d,
                     float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { sgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, &d, A, &ldA, B, &ldB, &scalfmm::S_ONE, C, &ldC); }
  inline void c_gemmta(int m, int p, int n, float *d,
                       float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, as_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ONE, as_c(C), &ldC); }
  inline void c_gemmta(int m, int p, int n, double *d,
                       double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+1, &m, &n, &p, as_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ONE, as_z(C), &ldC); }
  inline void c_gemmha(int m, int p, int n, float *d, // hermitian transposed
                       float *A, int ldA, float *B, int ldB, float *C, int ldC)
  { cgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+7, &m, &n, &p, as_c(d), as_c(A), &ldA, as_c(B), &ldB, &scalfmm::C_ONE, as_c(C), &ldC); }
  inline void c_gemmha(int m, int p, int n, double *d, // hermitian transposed
                       double *A, int ldA, double *B, int ldB, double *C, int ldC)
  { zgemm(scalfmm::JOB_STR, scalfmm::JOB_STR+7, &m, &n, &p, as_z(d), as_z(A), &ldA, as_z(B), &ldB, &scalfmm::Z_ONE, as_z(C), &ldC); }


  // singular value decomposition
    //
  inline int gesvd(int m, int n, double *A, double *S, double *VT, int ldVT,
                   int nwk, double *wk)
  {
    int INF;
    dgesvd(scalfmm::JOB_STR+2, scalfmm::JOB_STR+3, &m, &n, A, &m, S, A, &m, VT, &ldVT, wk, &nwk, &INF);
    return INF;
  }
  //
  //    A = U * SIGMA * conjugate-transpose(V)
  // scalfmm::JOB_STR+2 = 'O':  the first min(m,n) columns of U (the left singular vectors) are overwritten on the array A;
  inline int c_gesvd(int m, int n, double *A, double *S, double *VT, int ldVT,
                     int& nwk, double *wk,double *rwk)
  {
    int INF;
    zgesvd(scalfmm::JOB_STR+2, scalfmm::JOB_STR+3, &m, &n, as_z(A), &m, S, as_z(A), &m, as_z(VT), &ldVT, as_z(wk), &nwk, rwk, &INF);
    return INF;
  }
  inline int gesvd(int m, int n, float *A, float *S, float *VT, int ldVT,
                   int nwk, float *wk)
  {
    int INF;
    sgesvd(scalfmm::JOB_STR+2, scalfmm::JOB_STR+3, &m, &n, A, &m, S, A, &m, VT, &ldVT, wk, &nwk, &INF);
    return INF;
  }

  // singular value decomposition (SO)
  inline int gesvdSO(int m, int n, double *A, double *S, double *U, int ldU,
                     int nwk, double *wk)
  {
    int INF;
    dgesvd(scalfmm::JOB_STR+3, scalfmm::JOB_STR+2, &m, &n, A, &m, S, U, &m, A, &ldU, wk, &nwk, &INF);
    return INF;
  }
  inline int gesvdSO(int m, int n, float *A, float *S, float *U, int ldU,
                     int nwk, float *wk)
  {
    int INF;
    sgesvd(scalfmm::JOB_STR+3, scalfmm::JOB_STR+2, &m, &n, A, &m, S, U, &m, A, &ldU, wk, &nwk, &INF);
    return INF;
  }

  // singular value decomposition (AA)
  inline int gesvdAA(int m, int n, double *A, double *S, double *U, int ldU,
                     int nwk, double *wk)
  {
    int INF;
    dgesvd("A", "A", &m, &n, A, &m, S, U, &m, A, &ldU, wk, &nwk, &INF);
    return INF;
  }
  inline int gesvdAA(int m, int n, float *A, float *S, float *U, int ldU,
                     int nwk, float *wk)
  {
    int INF;
    sgesvd("A", "A", &m, &n, A, &m, S, U, &m, A, &ldU, wk, &nwk, &INF);
    return INF;
  }

  // Scalar product v1'*v2
  inline double scpr(int n, double *v1, double *v2)
  { return ddot(&n, v1, &scalfmm::N_ONE, v2, &scalfmm::N_ONE); }
  inline float scpr(int n, float *v1, float *v2)
  { return sdot(&n, v1, &scalfmm::N_ONE, v2, &scalfmm::N_ONE); }



  // QR factorisation
  inline int geqrf(const int m, const int n, double *A, double *tau, int nwk, double *wk)
  {
    int INF;
    dgeqrf(&m, &n, A, &m, tau, wk, &nwk, &INF);
    return INF;
  }
  inline int geqrf(const int m, const int n, float *A, float *tau, int nwk, float *wk)
  {
    int INF;
    sgeqrf(&m, &n, A, &m, tau, wk, &nwk, &INF);
    return INF;
  }
  inline int c_geqrf(const int m, const int n, float *A, float *tau, int nwk, float *wk)
  {
    int INF;
    cgeqrf(&m, &n, as_c(A), &m, as_c(tau), as_c(wk), &nwk, &INF);
    return INF;
  }
  inline int c_geqrf(const int m, const int n, double *A, double *tau, int nwk, double *wk)
  {
    int INF;
    zgeqrf(&m, &n, as_z(A), &m, as_z(tau), as_z(wk), &nwk, &INF);
    return INF;
  }

  // return full of Q-Matrix (QR factorization) in A
  inline int orgqr_full(const int m, const int n, double *A, double *tau, int nwk, double *wk)
  {
    int INF;
    dorgqr(&m, &m, &n, A, &m, tau, wk, &nwk, &INF);
    return INF;
  }
  inline int orgqr_full(const int m, const int n, float *A, float *tau, int nwk, float *wk)
  {
    int INF;
    sorgqr(&m, &m, &n, A, &m, tau, wk, &nwk, &INF);
    return INF;
  }
  // return the leading n columns of Q-Matrix (QR factorization) in A
  inline int orgqr(const int m, const int n, double *A, double *tau, int nwk, double *wk)
  {
    int INF;
    dorgqr(&m, &n, &n, A, &m, tau, wk, &nwk, &INF);
    return INF;
  }
  inline int orgqr(const int m, const int n, float *A, float *tau, int nwk, float *wk)
  {
    int INF;
    sorgqr(&m, &n, &n, A, &m, tau, wk, &nwk, &INF);
    return INF;
  }



  // apply Q-Matrix (from QR factorization) to C
  // LEFT: Q(^T)C
  inline int left_ormqr(const char* TRANS, const int m, const int n, const double *A, double *tau, double *C, int nwk, double *wk)
  {
    int INF;
    dormqr("L", TRANS, &m, &n, &m, A, &m, tau, C, &m, wk, &nwk, &INF);
    return INF;
  }
  inline int left_ormqr(const char* TRANS, const int m, const int n, const float *A, float *tau, float *C, int nwk, float *wk)
  {
    int INF;
    sormqr("L", TRANS, &m, &n, &m, A, &m, tau, C, &m, wk, &nwk, &INF);
    return INF;
  }
  // RIGHT: CQ(^T)
  inline int right_ormqr(const char* TRANS, const int m, const int n, const double *A, double *tau, double *C, int nwk, double *wk)
  {
    int INF;
    dormqr("R", TRANS, &m, &n, &n, A, &n, tau, C, &m, wk, &nwk, &INF);
    return INF;
  }
  inline int right_ormqr(const char* TRANS, const int m, const int n, const float *A, float *tau, float *C, int nwk, float *wk)
  {
    int INF;
    sormqr("R", TRANS, &m, &n, &n, A, &n, tau, C, &m, wk, &nwk, &INF);
    return INF;
  }

  // Cholesky decomposition: A=LL^T (if A is symmetric definite positive)
  inline int potrf(const int m, double *A, const int n)
  { 
    int INF;  
    dpotrf("L", &m, A, &n, &INF);
    return INF;
  }
  inline int potrf(const int m, float *A, const int n)
  { 
    int INF;  
    spotrf("L", &m, A, &n, &INF);
    return INF;
  }

} // end namespace FBlas

#undef as_c
#undef as_z
#undef as_const_c
#undef as_const_z

#endif //FBLAS_HPP

