// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <limits>
#include <memory>
#include <vector>

#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace rbf {
namespace reference {

class triharmonic3d final : public rbf_base {
public:
  using rbf_base::rbf_base;

  explicit triharmonic3d(const std::vector<double>& params) {
    set_parameters(params);
  }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<triharmonic3d>(*this);
  }

  int cpd_order() const override {
    return 2;
  }

  static double evaluate_untransformed(double r, const double *params) {
    auto slope = params[0];

    return slope * r * r * r;
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto slope = parameters()[0];

    auto c = 3.0 * slope * r;
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }

  int num_parameters() const override {
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

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
