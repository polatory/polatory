#pragma once

#include <polatory/fmm/gradient_kernel.hpp>
#include <polatory/fmm/gradient_transpose_kernel.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_gaussian.hpp>
#include <polatory/rbf/cov_generalized_cauchy3.hpp>
#include <polatory/rbf/cov_generalized_cauchy5.hpp>
#include <polatory/rbf/cov_generalized_cauchy7.hpp>
#include <polatory/rbf/cov_generalized_cauchy9.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/polyharmonic_even.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>

namespace polatory::fmm {

// Cost of one evaluation of the kernel on a pair of points relative to that of
// Kernel<Biharmonic3D<Dim>>: the geometric mean of the values measured by
// benchmark/kernel_cost.cpp on an Apple M-series Mac and on an x64 PC, which differ by up
// to a factor of 2.
template <class Kernel>
inline constexpr double kKernelCost = [] {
  static_assert(sizeof(Kernel) == 0, "the cost of the kernel has not been measured");
  return 0.0;
}();

#define POLATORY_KERNEL_COST(RBF, KERNEL, GRADIENT, GRADIENT_TRANSPOSE, HESSIAN)      \
  template <>                                                                         \
  inline constexpr double kKernelCost<Kernel<rbf::internal::RBF>> = KERNEL;           \
  template <>                                                                         \
  inline constexpr double kKernelCost<GradientKernel<rbf::internal::RBF>> = GRADIENT; \
  template <>                                                                         \
  inline constexpr double kKernelCost<GradientTransposeKernel<rbf::internal::RBF>> =  \
      GRADIENT_TRANSPOSE;                                                             \
  template <>                                                                         \
  inline constexpr double kKernelCost<HessianKernel<rbf::internal::RBF>> = HESSIAN;

POLATORY_KERNEL_COST(Biharmonic2D<1>, 3.0, 3.7, 3.8, 4.8)
POLATORY_KERNEL_COST(Biharmonic2D<2>, 3.7, 4.8, 4.7, 8.8)
POLATORY_KERNEL_COST(Biharmonic2D<3>, 3.7, 5.5, 5.5, 18.0)
POLATORY_KERNEL_COST(Biharmonic3D<1>, 1.0, 1.5, 1.6, 2.3)
POLATORY_KERNEL_COST(Biharmonic3D<2>, 1.0, 1.9, 1.9, 4.7)
POLATORY_KERNEL_COST(Biharmonic3D<3>, 1.0, 2.7, 2.7, 12.0)
POLATORY_KERNEL_COST(CovExponential<1>, 3.6, 4.3, 4.3, 5.3)
POLATORY_KERNEL_COST(CovExponential<2>, 3.7, 5.2, 5.3, 9.1)
POLATORY_KERNEL_COST(CovExponential<3>, 3.5, 6.3, 6.2, 18.0)
POLATORY_KERNEL_COST(CovGaussian<1>, 3.8, 4.5, 4.4, 4.9)
POLATORY_KERNEL_COST(CovGaussian<2>, 3.8, 5.4, 5.4, 8.5)
POLATORY_KERNEL_COST(CovGaussian<3>, 3.6, 6.4, 6.4, 17.0)
POLATORY_KERNEL_COST(CovGeneralizedCauchy3<1>, 2.7, 3.2, 3.2, 3.9)
POLATORY_KERNEL_COST(CovGeneralizedCauchy3<2>, 2.6, 3.8, 3.8, 6.7)
POLATORY_KERNEL_COST(CovGeneralizedCauchy3<3>, 2.2, 4.3, 4.3, 15.0)
POLATORY_KERNEL_COST(CovGeneralizedCauchy5<1>, 2.8, 3.3, 3.4, 4.3)
POLATORY_KERNEL_COST(CovGeneralizedCauchy5<2>, 2.6, 3.8, 3.9, 6.9)
POLATORY_KERNEL_COST(CovGeneralizedCauchy5<3>, 2.5, 4.3, 4.2, 15.0)
POLATORY_KERNEL_COST(CovGeneralizedCauchy7<1>, 2.8, 3.2, 3.2, 4.0)
POLATORY_KERNEL_COST(CovGeneralizedCauchy7<2>, 2.7, 3.9, 3.8, 7.1)
POLATORY_KERNEL_COST(CovGeneralizedCauchy7<3>, 2.6, 4.6, 4.6, 16.0)
POLATORY_KERNEL_COST(CovGeneralizedCauchy9<1>, 2.6, 3.2, 3.1, 3.8)
POLATORY_KERNEL_COST(CovGeneralizedCauchy9<2>, 2.6, 3.8, 3.7, 7.0)
POLATORY_KERNEL_COST(CovGeneralizedCauchy9<3>, 2.4, 4.4, 4.5, 15.0)
POLATORY_KERNEL_COST(CovSpheroidal3FastPart<1>, 2.6, 3.1, 3.0, 3.9)
POLATORY_KERNEL_COST(CovSpheroidal3FastPart<2>, 2.6, 3.8, 3.8, 6.9)
POLATORY_KERNEL_COST(CovSpheroidal3FastPart<3>, 2.4, 4.5, 4.5, 16.0)
POLATORY_KERNEL_COST(CovSpheroidal5FastPart<1>, 2.7, 3.3, 3.2, 4.0)
POLATORY_KERNEL_COST(CovSpheroidal5FastPart<2>, 2.7, 3.9, 3.9, 7.1)
POLATORY_KERNEL_COST(CovSpheroidal5FastPart<3>, 2.5, 4.5, 4.6, 16.0)
POLATORY_KERNEL_COST(CovSpheroidal7FastPart<1>, 2.8, 3.1, 3.2, 4.0)
POLATORY_KERNEL_COST(CovSpheroidal7FastPart<2>, 2.8, 3.9, 3.8, 7.0)
POLATORY_KERNEL_COST(CovSpheroidal7FastPart<3>, 2.6, 4.6, 4.5, 16.0)
POLATORY_KERNEL_COST(CovSpheroidal9FastPart<1>, 2.6, 3.2, 3.1, 3.9)
POLATORY_KERNEL_COST(CovSpheroidal9FastPart<2>, 2.6, 3.7, 3.7, 7.0)
POLATORY_KERNEL_COST(CovSpheroidal9FastPart<3>, 2.4, 4.4, 4.4, 15.0)
POLATORY_KERNEL_COST(Triharmonic2D<1>, 3.9, 4.3, 4.3, 5.3)
POLATORY_KERNEL_COST(Triharmonic2D<2>, 4.0, 5.2, 5.3, 9.8)
POLATORY_KERNEL_COST(Triharmonic2D<3>, 4.0, 6.1, 6.1, 20.0)
POLATORY_KERNEL_COST(Triharmonic3D<1>, 1.1, 1.1, 1.1, 1.9)
POLATORY_KERNEL_COST(Triharmonic3D<2>, 1.1, 1.6, 1.6, 4.5)
POLATORY_KERNEL_COST(Triharmonic3D<3>, 1.1, 2.5, 2.4, 12.0)

#undef POLATORY_KERNEL_COST

}  // namespace polatory::fmm
