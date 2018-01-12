// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <memory>

#include <polatory/rbf/rbf_kernel.hpp>

namespace polatory {
namespace rbf {

class biharmonic2d : public rbf_kernel {
public:
  using rbf_kernel::rbf_kernel;

  std::shared_ptr<rbf_kernel> clone() const override {
    return std::make_shared<biharmonic2d>(parameters());
  }

  int cpd_order() const override {
    return 2;
  }

  static double evaluate(double r, const double *params) {
    auto slope = params[0];

    return r == 0.0
           ? 0.0
           : slope * r * r * std::log(r);
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
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

  double nugget() const override {
    return parameters()[1];
  }
};

}  // namespace rbf
}  // namespace polatory
