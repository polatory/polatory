// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace rbf {
namespace reference {

class triharmonic final : public rbf_base {
public:
  using rbf_base::rbf_base;

  explicit triharmonic(const std::vector<double>& params) {
    set_parameters(params);
  }

  triharmonic(const triharmonic& other) {
    set_parameters(other.parameters());
  }

  std::shared_ptr<rbf_base> clone() const override {
    return std::make_shared<triharmonic>(parameters());
  }

  int cpd_order() const override {
    return 2;
  }

  static double evaluate(double r, const double *params) {
    auto slope = params[0];

    return slope * r * r * r;
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto slope = parameters()[0];

    auto c = 3.0 * slope * r;
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }

  double nugget() const override {
    return parameters()[1];
  }

  size_t num_parameters() const override {
    return 2;
  }
};

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
