// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <polatory/rbf/covariance_function.hpp>

namespace polatory {
namespace rbf {

class cov_quasi_spherical9 : public covariance_function {
public:
  using covariance_function::covariance_function;

  std::shared_ptr<rbf_kernel> clone() const override {
    return std::make_shared<cov_quasi_spherical9>(parameters());
  }

  static double evaluate(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < 0.3211688011376158 * range
           ? psill * (1.0 - 1.4011323590773752 * r / range)
           : psill * 0.9710239190254878 * std::pow(1.031493988241734 + std::pow(r / range, 2.0), -4.5);
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = r < 0.3211688011376158 * range
             ? -psill * 1.4011323590773752 / (range * r)
             : -psill * 8.73921527122939 * std::pow(1.031493988241734 + std::pow(r / range, 2.0), -5.5) / (range * range);
    gradx = c * x;
    grady = c * y;
    gradz = c * z;
  }

  POLATORY_DEFINE_COST_FUNCTION(cov_quasi_spherical9, 3)
};

} // namespace rbf
} // namespace polatory
