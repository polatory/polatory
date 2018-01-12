// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/rbf/rbf_kernel.hpp>

namespace polatory {
namespace rbf {
namespace reference {

class triharmonic : public rbf_kernel {
public:
  using rbf_kernel::rbf_kernel;

  std::shared_ptr<rbf_kernel> clone() const override {
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
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto slope = parameters()[0];

    auto c = 3.0 * slope * r;
    gradx = c * x;
    grady = c * y;
    gradz = c * z;
  }

  double nugget() const override {
    return parameters()[1];
  }
};

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
