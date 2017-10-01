// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <limits>

#include "variogram.hpp"

namespace polatory {
namespace rbf {

struct linear_variogram : variogram {
  using variogram::variogram;

  static double evaluate(double r, const double *params) {
    auto slope = params[0];
    auto nugget = params[1];

    return slope * r + nugget;
  }

  double evaluate(double r) const override {
    return evaluate(r, parameters());
  }

  void evaluate_gradient(
    double& gradx, double& grady, double& gradz,
    double x, double y, double z, double r) const override {
    auto slope = parameters()[0];

    auto c = slope / r;
    gradx = c * x;
    grady = c * y;
    gradz = c * z;
  }

  double nugget() const override {
    return parameters()[1];
  }

  const double *parameter_lower_bounds() const override {
    static const double lower_bounds[]{ 0.0, 0.0 };
    return lower_bounds;
  }

  const double *parameter_upper_bounds() const override {
    static const double upper_bounds[]{ std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity() };
    return upper_bounds;
  }

  int num_parameters() const override {
    return 2;
  }

  DECLARE_COST_FUNCTIONS(linear_variogram)
};

DEFINE_COST_FUNCTIONS(linear_variogram, 2)

} // namespace rbf
} // namespace polatory
