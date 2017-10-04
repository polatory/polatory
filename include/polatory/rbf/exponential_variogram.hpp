// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>

#include "covariance_function.hpp"

namespace polatory {
namespace rbf {

class exponential_variogram : public covariance_function {
public:
  using covariance_function::covariance_function;

  static double evaluate(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return psill * std::exp(-r / range);
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = -psill * std::exp(-r / range) / (range * r);
    gradx = c * x;
    grady = c * y;
    gradz = c * z;
  }

  DECLARE_COST_FUNCTIONS(exponential_variogram)
};

DEFINE_COST_FUNCTIONS(exponential_variogram, 3)

} // namespace rbf
} // namespace polatory
