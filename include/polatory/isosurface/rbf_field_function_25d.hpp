// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include "field_function.hpp"
#include "polatory/geometry/bbox3d.hpp"
#include "polatory/interpolant.hpp"

namespace polatory {
namespace isosurface {

struct rbf_field_function_25d : field_function {
  rbf_field_function_25d(interpolant& interpolant)
    : interpolant_(interpolant) {
  }

  Eigen::VectorXd operator()(const std::vector<Eigen::Vector3d>& points) const override {
    std::vector<Eigen::Vector3d> points_2d(points);
    for (auto& p : points_2d) {
      p(2) = 0.0;
    }

    auto values = interpolant_.evaluate_points_impl(points_2d);
    for (size_t i = 0; i < points.size(); i++) {
      values(i) = points[i](2) - values(i);
    }

    return values;
  }

  void set_evaluation_bbox(const geometry::bbox3d& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox);
  }

private:
  interpolant& interpolant_;
};

} // namespace isosurface
} // namespace polatory
