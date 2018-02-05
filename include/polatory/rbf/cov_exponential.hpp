// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <polatory/rbf/covariance_function_base.hpp>

namespace polatory {
namespace rbf {

class cov_exponential : public covariance_function_base {
public:
  using covariance_function_base::covariance_function_base;

  std::shared_ptr<rbf_base> clone() const override {
    return std::make_shared<cov_exponential>(parameters());
  }

  static double evaluate(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return psill * std::exp(-r / range);
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = -psill * std::exp(-r / range) / (range * r);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }

  POLATORY_DEFINE_COST_FUNCTION(cov_exponential, 3)
};

}  // namespace rbf
}  // namespace polatory
