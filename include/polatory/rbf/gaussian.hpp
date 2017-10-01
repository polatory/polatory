// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>

#include "rbf_base.hpp"

namespace polatory {
namespace rbf {

struct gaussian : rbf_base {
  using rbf_base::rbf_base;

  static double evaluate(double r, const double *params) {
    auto c = params[0];

    return std::exp(-r * r / (c * c));
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters());
  }

  void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto c = parameters()[0];
    auto d = -2.0 * std::exp(-r * r / (c * c)) / (c * c);

    gradx = d * x;
    grady = d * y;
    gradz = d * z;
  }

  int definiteness() const override {
    return 1;
  }

  int order_of_definiteness() const override {
    return 0;
  }
};

} // namespace rbf
} // namespace polatory
