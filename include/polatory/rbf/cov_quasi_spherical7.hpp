// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <polatory/rbf/covariance_function_base.hpp>

namespace polatory {
namespace rbf {

class cov_quasi_spherical7 : public covariance_function_base {
public:
  using covariance_function_base::covariance_function_base;

  std::shared_ptr<rbf_base> clone() const override {
    return std::make_shared<cov_quasi_spherical7>(parameters());
  }

  static double evaluate(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < 0.2994221418316404 * range
           ? psill * (1.0 - 1.4611477872802014 * r / range)
           : psill * 0.2654353119171418 * std::pow(0.7172289521523758 + std::pow(r / range, 2.0), -3.5);
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = r < 0.2994221418316404 * range
             ? -psill * 1.4611477872802014 / (range * r)
             : -psill * 1.8580471834199928 * std::pow(0.7172289521523758 + std::pow(r / range, 2.0), -4.5) / (range * range);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }

  POLATORY_DEFINE_COST_FUNCTION(cov_quasi_spherical7, 3)
};

}  // namespace rbf
}  // namespace polatory
