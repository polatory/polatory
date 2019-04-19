// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <polatory/rbf/covariance_function_base.hpp>

namespace polatory {
namespace rbf {
namespace reference {

class cov_spherical : public covariance_function_base {
public:
  using covariance_function_base::covariance_function_base;

  std::shared_ptr<rbf_base> clone() const override {
    return std::make_shared<cov_spherical>(parameters());
  }

  static double evaluate(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < range
           ? psill * (1.0 - 1.5 * r / range + 0.5 * std::pow(r / range, 3.0))
           : 0.0;
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    if (r < range) {
      auto c = psill * 1.5 * (-1.0 / (range * r) + r / std::pow(range, 3.0));
      *gradx = c * x;
      *grady = c * y;
      *gradz = c * z;
    } else {
      *gradx = 0.0;
      *grady = 0.0;
      *gradz = 0.0;
    }
  }
};

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
