// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>

#include "rbf_base.hpp"

namespace polatory {
namespace rbf {

class biharmonic2d : public rbf_base {
public:
  using rbf_base::rbf_base;

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
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto slope = parameters()[0];

    if (r == 0.0) {
      gradx = 0.0;
      grady = 0.0;
      gradz = 0.0;
    } else {
      auto c = slope * (1.0 + 2.0 * std::log(r));
      gradx = c * x;
      grady = c * y;
      gradz = c * z;
    }
  }

  double nugget() const override {
    return parameters()[1];
  }

  int order_of_cpd() const override {
    return 2;
  }
};

} // namespace rbf
} // namespace polatory
