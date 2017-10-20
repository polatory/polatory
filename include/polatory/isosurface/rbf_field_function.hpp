// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include "field_function.hpp"
#include "polatory/interpolant.hpp"

namespace polatory {
namespace isosurface {

struct rbf_field_function : field_function {
  rbf_field_function(const interpolant& interpolant)
    : interpolant_(interpolant) {
  }

  Eigen::VectorXd operator()(const std::vector<Eigen::Vector3d>& points) const override {
    return interpolant_.evaluate_points_impl(points);
  }

private:
  const interpolant& interpolant_;
};

} // namespace isosurface
} // namespace polatory
