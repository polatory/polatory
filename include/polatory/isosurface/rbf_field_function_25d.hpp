// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/field_function.hpp>

namespace polatory {
namespace isosurface {

struct rbf_field_function_25d : field_function {
  rbf_field_function_25d(interpolant& interpolant)
    : interpolant_(interpolant) {
  }

  common::valuesd operator()(const geometry::points3d& points) const override {
    geometry::points3d points_2d(points);
    points_2d.col(2).array() = 0.0;

    return points.col(2) -
      interpolant_.evaluate_points_impl(points_2d);
  }

  void set_evaluation_bbox(const geometry::bbox3d& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox);
  }

private:
  interpolant& interpolant_;
};

} // namespace isosurface
} // namespace polatory
