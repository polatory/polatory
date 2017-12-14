// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <polatory/rbf/covariance_function.hpp>

namespace polatory {
namespace rbf {

class cov_gaussian : public covariance_function {
public:
  using covariance_function::covariance_function;

  std::shared_ptr<rbf_kernel> clone() const override {
    return std::make_shared<cov_gaussian>(parameters());
  }

  static double evaluate(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return psill * std::exp(-r * r / (range * range));
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = -2.0 * psill * std::exp(-r * r / (range * range)) / (range * range);
    gradx = c * x;
    grady = c * y;
    gradz = c * z;
  }

  POLATORY_DEFINE_COST_FUNCTIONS(cov_gaussian, 3)
};

} // namespace rbf
} // namespace polatory
