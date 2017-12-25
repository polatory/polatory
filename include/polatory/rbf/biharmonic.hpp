// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/rbf/rbf_kernel.hpp>

namespace polatory {
namespace rbf {

class biharmonic : public rbf_kernel {
public:
  using rbf_kernel::rbf_kernel;

  std::shared_ptr<rbf_kernel> clone() const override {
    return std::make_shared<biharmonic>(parameters());
  }

  static double evaluate(double r, const double *params) {
    auto slope = params[0];

    return -slope * r;
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters().data());
  }

  void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto slope = parameters()[0];

    auto c = -slope / r;
    gradx = c * x;
    grady = c * y;
    gradz = c * z;
  }

  double nugget() const override {
    return parameters()[1];
  }

  int order_of_cpd() const override {
    return 1;
  }
};

}  // namespace rbf
}  // namespace polatory
