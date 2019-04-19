// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include <polatory/rbf/covariance_function_base.hpp>

namespace polatory {
namespace rbf {

class cov_exponential final : public covariance_function_base {
public:
  using covariance_function_base::covariance_function_base;

  explicit cov_exponential(const std::vector<double>& params) {
    set_parameters(params);
  }

  cov_exponential(const cov_exponential& other) {
    set_parameters(other.parameters());
  }

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
};

}  // namespace rbf
}  // namespace polatory
