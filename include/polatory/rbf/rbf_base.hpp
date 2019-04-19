// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <polatory/common/exception.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/third_party/ScalFMM/Kernels/Interpolation/FInterpMatrixKernel.hpp>

namespace polatory {
namespace rbf {

class rbf_base : FInterpAbstractMatrixKernel<double> {
public:
  rbf_base() = default;

  virtual std::shared_ptr<rbf_base> clone() const = 0;

  // The order of conditional positive definiteness.
  virtual int cpd_order() const = 0;

  virtual double evaluate(double r) const = 0;

  virtual void evaluate_gradient(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const = 0;

  // The effect of nugget parameter is also known as spline smoothing.
  virtual double nugget() const {
    return 0.0;
  }

  virtual size_t num_parameters() const = 0;

  const std::vector<double>& parameters() const {
    return params_;
  }

  void set_parameters(const std::vector<double>& params) {
    if (params.size() != num_parameters())
      throw common::invalid_argument("params.size() == " + std::to_string(num_parameters()));

    params_ = params;
  }

  // The following definitions are used in ScalFMM.

  static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
  static const unsigned int NCMP = 1;    // Number of components.
  static const unsigned int NPV = 1;     // Dimension of physical values.
  static const unsigned int NPOT = 1;    // Dimension of potentials.
  static const unsigned int NRHS = 1;    // Dimension of multipole expansions.
  static const unsigned int NLHS = 1;    // Dimension of local expansions.

  // returns position in reduced storage
  int getPosition(const unsigned int) const {
    return 0;
  }

  double getMutualCoefficient() const {
    return 1.0;
  }

  // evaluate interaction
  double evaluate(const double *p1, const double *p2) const {
    const auto diffx = p1[0] - p2[0];
    const auto diffy = p1[1] - p2[1];
    const auto diffz = p1[2] - p2[2];
    const auto r = std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);

    return evaluate(r);
  }

  // evaluate interaction and derivative (blockwise)
  void evaluateBlockAndDerivative(const double& x1, const double& y1, const double& z1,
                                  const double& x2, const double& y2, const double& z2,
                                  double block[1], double blockDerivative[3]) const {
    const auto diffx = (x1 - x2);
    const auto diffy = (y1 - y2);
    const auto diffz = (z1 - z2);
    const auto r = std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);

    block[0] = evaluate(r);
    evaluate_gradient(&blockDerivative[0], &blockDerivative[1], &blockDerivative[2], diffx, diffy, diffz, r);
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

  double evaluate(const geometry::point3d& p1, const geometry::point3d& p2) const {
    return evaluate(p1.data(), p2.data());
  }

private:
  std::vector<double> params_;
};

}  // namespace rbf
}  // namespace polatory
