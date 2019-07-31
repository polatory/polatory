// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include <polatory/rbf/covariance_function_base.hpp>

namespace polatory {
namespace rbf {

class cov_quasi_spherical5 final : public covariance_function_base {
public:
  using covariance_function_base::covariance_function_base;

  explicit cov_quasi_spherical5(const std::vector<double>& params) {
    set_parameters(params);
  }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_quasi_spherical5>(*this);
  }

  static double evaluate_untransformed(double r, const double *params) {
    auto psill = params[0];

    return r < 0.2630893471886411
           ? psill * (1.0 - 1.5837458685391284 * r)
           : psill * 0.09531872195083474 * std::pow(0.415296027624872 + std::pow(r, 2.0), -2.5);
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];

    auto c = r < 0.2630893471886411
             ? -psill * 1.5837458685391284 / r
             : -psill * 0.4765936097541737 * std::pow(0.415296027624872 + std::pow(r, 2.0), -3.5);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }
};

}  // namespace rbf
}  // namespace polatory
