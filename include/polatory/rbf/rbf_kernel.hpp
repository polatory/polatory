// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include <ceres/ceres.h>
#include <Eigen/Core>

#include <polatory/common/exception.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/third_party/ScalFMM/Kernels/Interpolation/FInterpMatrixKernel.hpp>

namespace polatory {
namespace rbf {

class rbf_kernel : FInterpAbstractMatrixKernel<double> {
public:
  explicit rbf_kernel(const std::vector<double>& params)
    : params_(params) {
  }

  rbf_kernel(const rbf_kernel& other)
    : params_(other.params_) {
  }

  virtual std::shared_ptr<rbf_kernel> clone() const = 0;

  virtual double evaluate(double r) const = 0;

  virtual void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const = 0;

  // The effect of nugget parameter is also known as spline smoothing.
  virtual double nugget() const {
    return 0.0;
  }

  // The order of conditional positive definiteness.
  virtual int order_of_cpd() const = 0;

  const std::vector<double>& parameters() const {
    return params_;
  }

  // The following members are used in covariance functions.

  virtual int num_parameters() const {
    throw common::not_supported("num_parameters");
  }

  virtual const double *parameter_lower_bounds() const {
    throw common::not_supported("parameter_lower_bounds");
  }

  virtual const double *parameter_upper_bounds() const {
    throw common::not_supported("parameter_upper_bounds");
  }

  virtual ceres::CostFunction *cost_function(double h, double gamma, double weight) const {
    throw common::not_supported("cost_function");
  }

  virtual ceres::CostFunction *cost_function_over_gamma(double h, double gamma, double weight) const {
    throw common::not_supported("cost_function_over_gamma");
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
    evaluate_gradient(blockDerivative[0], blockDerivative[1], blockDerivative[2], diffx, diffy, diffz, r);
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
  const std::vector<double> params_;
};

} // namespace rbf
} // namespace polatory
