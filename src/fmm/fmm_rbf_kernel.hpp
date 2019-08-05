// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <ScalFMM/Kernels/Interpolation/FInterpMatrixKernel.hpp>

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

  explicit fmm_rbf_kernel(const rbf::rbf_base& rbf)
    : rbf_(rbf.clone()) {
  }

  // returns position in reduced storage
  int getPosition(const unsigned int) const {
    return 0;
  }

  double getMutualCoefficient() const {
    return 1.0;
  }

  // evaluate interaction
  double evaluate(const double* p1, const double* p2) const {
    const auto diffx = p1[0] - p2[0];
    const auto diffy = p1[1] - p2[1];
    const auto diffz = p1[2] - p2[2];
    const auto r = std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);

    return rbf_->evaluate_untransformed(r);
  }

  // evaluate interaction and derivative (blockwise)
  void evaluateBlockAndDerivative(const double& x1, const double& y1, const double& z1,
    const double& x2, const double& y2, const double& z2,
    double block[1], double blockDerivative[3]) const {
    const auto diffx = (x1 - x2);
    const auto diffy = (y1 - y2);
    const auto diffz = (z1 - z2);
    const auto r = std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);

    block[0] = rbf_->evaluate_untransformed(r);

    if (kEvaluateGradient) {
      rbf_->evaluate_gradient_untransformed(&blockDerivative[0], &blockDerivative[1], &blockDerivative[2], diffx, diffy, diffz, r);
    } else {
      blockDerivative[0] = 0.0;
      blockDerivative[1] = 0.0;
      blockDerivative[2] = 0.0;
    }
  }

  double getScaleFactor(const double, const int) const override {
    // return 1 because non homogeneous kernel functions cannot be scaled!!!
    return 1.0;
  }

  double getScaleFactor(const double) const override {
    // return 1 because non homogeneous kernel functions cannot be scaled!!!
    return 1.0;
  }

  double evaluate(const FPoint<double>& p1, const FPoint<double>& p2) const {
    return evaluate(p1.data(), p2.data());
  }

private:
  const std::unique_ptr<rbf::rbf_base> rbf_;
};

}  // namespace fmm
}  // namespace polatory
