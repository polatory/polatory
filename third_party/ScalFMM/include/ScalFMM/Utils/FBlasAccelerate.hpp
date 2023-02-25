// See LICENCE file at project root
#ifndef FBLASACCELERATE_HPP
#define FBLASACCELERATE_HPP

extern "C" {
// double //////////////////////////////////////////////////////////
// blas 1
double ddot(const int *, const double *, const int *, const double *, const int *);
void dscal(const int *, const double *, const double *, const int *);
void dcopy(const int *, const double *, const int *, double *, const int *);
void daxpy(const int *, const double *, const double *, const int *, double *, const int *);
// blas 2
void dgemv(const char *, const int *, const int *, const double *, const double *, const int *,
           const double *, const int *, const double *, double *, const int *);
// blas 3
void dgemm(const char *, const char *, const int *, const int *, const int *, const double *,
           double *, const int *, double *, const int *, const double *, double *, const int *);
// lapack
void dgesvd(const char *, const char *, const int *, const int *, double *, const int *, double *,
            double *, const int *, double *, const int *, double *, const int *, int *);
void dgeqrf(const int *, const int *, double *, const int *, double *, double *, const int *,
            int *);
void dgeqp3(const int *, const int *, double *, const int *, /*TYPE OF JPIV*/ int *, double *,
            double *, const int *, int *);
void dorgqr(const int *, const int *, const int *, double *, const int *, double *, double *,
            const int *, int *);
void dormqr(const char *, const char *, const int *, const int *, const int *, const double *,
            const int *, double *, double *, const int *, double *, const int *, int *);
void dpotrf(const char *, const int *, double *, const int *, int *);

// single //////////////////////////////////////////////////////////
// blas 1
float sdot(const int *, const float *, const int *, const float *, const int *);
void sscal(const int *, const float *, const float *, const int *);
void scopy(const int *, const float *, const int *, float *, const int *);
void saxpy(const int *, const float *, const float *, const int *, float *, const int *);
// blas 2
void sgemv(const char *, const int *, const int *, const float *, const float *, const int *,
           const float *, const int *, const float *, float *, const int *);
// blas 3
void sgemm(const char *, const char *, const int *, const int *, const int *, const float *,
           float *, const int *, float *, const int *, const float *, float *, const int *);
// lapack
void sgesvd(const char *, const char *, const int *, const int *, float *, const int *, float *,
            float *, const int *, float *, const int *, float *, const int *, int *);
void sgeqrf(const int *, const int *, float *, const int *, float *, float *, const int *, int *);
void sgeqp3(const int *, const int *, float *, const int *, /*TYPE OF JPIV*/ int *, float *,
            float *, const int *, int *);
void sorgqr(const int *, const int *, const int *, float *, const int *, float *, float *,
            const int *, int *);
void sormqr(const char *, const char *, const int *, const int *, const int *, const float *,
            const int *, float *, float *, const int *, float *, const int *, int *);
void spotrf(const char *, const int *, float *, const int *, int *);

// double complex //////////////////////////////////////////////////
// blas 1
void zscal(const int *, const double *, const double *, const int *);
void zcopy(const int *, const double *, const int *, double *, const int *);
void zaxpy(const int *, const double *, const double *, const int *, double *, const int *);
// blas 2
void zgemv(const char *, const int *, const int *, const double *, const double *, const int *,
           const double *, const int *, const double *, double *, const int *);
// blas 3
void zgemm(const char *, const char *, const int *, const int *, const int *, const double *,
           double *, const int *, double *, const int *, const double *, double *, const int *);
// lapack
void zgesvd(const char *, const char *, const int *, const int *, double *, const int *, double *,
            double *, const int *, double *, const int *, double *, int *, double *, int *);
void zgeqrf(const int *, const int *, double *, const int *, double *, double *, const int *,
            int *);
void zgeqp3(const int *, const int *, double *, const int *, /*TYPE OF JPIV*/ int *, double *,
            double *, const int *, int *);
void zpotrf(const char *, const int *, double *, const int *, int *);

// single complex //////////////////////////////////////////////////
// blas 1
void cscal(const int *, const float *, const float *, const int *);
void ccopy(const int *, const float *, const int *, float *, const int *);
void caxpy(const int *, const float *, const float *, const int *, float *, const int *);
// blas 2
void cgemv(const char *, const int *, const int *, const float *, const float *, const int *,
           const float *, const int *, const float *, float *, const int *);
// blas 3
void cgemm(const char *, const char *, const int *, const int *, const int *, const float *,
           float *, const int *, float *, const int *, const float *, float *, const int *);
// lapack
void cgeqrf(const int *, const int *, float *, const int *, float *, float *, const int *, int *);
void cgeqp3(const int *, const int *, float *, const int *, /*TYPE OF JPIV*/ int *, float *,
            float *, const int *, int *);
void cpotrf(const char *, const int *, float *, const int *, int *);
}

#define as_c(x) reinterpret_cast<float *>(x)
#define as_z(x) reinterpret_cast<double *>(x)
#define as_const_c(x) reinterpret_cast<const float *>(x)
#define as_const_z(x) reinterpret_cast<const double *>(x)

#endif  // FBLASACCELERATE_HPP
