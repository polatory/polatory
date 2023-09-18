#pragma once

#include <ScalFMM/Kernels/Interpolation/FInterpMatrixKernel.hpp>
#include <cmath>
#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace fmm {

struct fmm_rbf_kernel : FInterpAbstractMatrixKernel<double> {
  static constexpr bool kEvaluateGradient = false;

  static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
  static const unsigned int NCMP = 1;  // Number of components.
  static const unsigned int NPV = 1;   // Dimension of physical values.
  static const unsigned int NPOT = 1;  // Dimension of potentials.
  static const unsigned int NRHS = 1;  // Dimension of multipole expansions.
  static const unsigned int NLHS = 1;  // Dimension of local expansions.

  explicit fmm_rbf_kernel(const rbf::rbf_base& rbf) : rbf_(rbf.clone()) {}

  // returns position in reduced storage
  int getPosition(const unsigned int) const { return 0; }

  double getMutualCoefficient() const {
    // The kernel is symmetric.
    return 1.0;
  }

  // evaluate interaction
  double evaluate(double xt, double yt, double zt, double xs, double ys, double zs) const {
    geometry::vector3d diff{xt - xs, yt - ys, zt - zs};

    return rbf_->evaluate_untransformed(diff);
  }

  // evaluate interaction and derivative (blockwise)
  void evaluateBlockAndDerivative(double xt, double yt, double zt, double xs, double ys, double zs,
                                  double block[1], double blockDerivative[3]) const {
    geometry::vector3d diff{xt - xs, yt - ys, zt - zs};

    block[0] = rbf_->evaluate_untransformed(diff);

    if (kEvaluateGradient) {
      auto grad = rbf_->evaluate_gradient_untransformed(diff);
      blockDerivative[0] = grad(0);
      blockDerivative[1] = grad(1);
      blockDerivative[2] = grad(2);
    } else {
      blockDerivative[0] = 0.0;
      blockDerivative[1] = 0.0;
      blockDerivative[2] = 0.0;
    }
  }

  double getScaleFactor(const double, const int) const override {
    // The kernel is not homogeneous.
    return 1.0;
  }

  double getScaleFactor(const double) const override {
    // The kernel is not homogeneous.
    return 1.0;
  }

  double evaluate(const FPoint<double>& pt, const FPoint<double>& ps) const {
    return evaluate(pt.getX(), pt.getY(), pt.getZ(), ps.getX(), ps.getY(), ps.getZ());
  }

 private:
  const std::unique_ptr<rbf::rbf_base> rbf_;
};

}  // namespace fmm
}  // namespace polatory
