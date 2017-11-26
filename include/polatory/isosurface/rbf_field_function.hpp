// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/isosurface/field_function.hpp>

namespace polatory {
namespace isosurface {

struct rbf_field_function : field_function {
  rbf_field_function(interpolant& interpolant)
    : interpolant_(interpolant) {
  }

  Eigen::VectorXd operator()(const geometry::points3d& points) const override {
    return interpolant_.evaluate_points_impl(points);
  }

  void set_evaluation_bbox(const geometry::bbox3d& bbox) override {
    interpolant_.set_evaluation_bbox_impl(bbox);
  }

private:
  interpolant& interpolant_;
};

} // namespace isosurface
} // namespace polatory
