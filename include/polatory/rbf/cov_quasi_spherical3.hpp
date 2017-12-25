// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <polatory/rbf/covariance_function.hpp>

namespace polatory {
namespace rbf {

class cov_quasi_spherical3 : public covariance_function {
public:
  using covariance_function::covariance_function;

  std::shared_ptr<rbf_kernel> clone() const override {
    return std::make_shared<cov_quasi_spherical3>(parameters());
  }

  static double evaluate(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < 0.19156525704423027 * range
           ? psill * (1.0 - 1.9575574704207284 * r / range)
           : psill * 0.04912304321996779 * std::pow(0.14678899082568805 + std::pow(r / range, 2.0), -1.5);
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = r < 0.19156525704423027 * range
             ? -psill * 1.9575574704207284 / (range * r)
             : -psill * 0.14736912965990337 * std::pow(0.14678899082568805 + std::pow(r / range, 2.0), -2.5) / (range * range);
    gradx = c * x;
    grady = c * y;
    gradz = c * z;
  }

  POLATORY_DEFINE_COST_FUNCTION(cov_quasi_spherical3, 3)
};

}  // namespace rbf
}  // namespace polatory
