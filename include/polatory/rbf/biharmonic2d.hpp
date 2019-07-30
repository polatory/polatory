// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace rbf {

class biharmonic2d final : public rbf_base {
public:
  using rbf_base::rbf_base;

  explicit biharmonic2d(const std::vector<double>& params) {
    set_parameters(params);
  }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<biharmonic2d>(*this);
  }

  int cpd_order() const override {
    return 2;
  }

  static double evaluate_untransformed(double r, const double *params) {
    auto slope = params[0];

    return r == 0.0
           ? 0.0
           : slope * r * r * std::log(r);
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto slope = parameters()[0];

    if (r == 0.0) {
      *gradx = 0.0;
      *grady = 0.0;
      *gradz = 0.0;
    } else {
      auto c = slope * (1.0 + 2.0 * std::log(r));
      *gradx = c * x;
      *grady = c * y;
      *gradz = c * z;
    }
  }

  size_t num_parameters() const override {
    return 1;
  }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{ 0.0 };
    return lower_bounds;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{ std::numeric_limits<double>::infinity() };
    return upper_bounds;
  }
};

}  // namespace rbf
}  // namespace polatory
