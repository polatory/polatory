// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

#include <polatory/geometry/bbox3d.hpp>

namespace polatory {
namespace isosurface {

struct field_function {
  virtual ~field_function() {}

  virtual Eigen::VectorXd operator()(const std::vector<Eigen::Vector3d>& points) const = 0;

  virtual void set_evaluation_bbox(const geometry::bbox3d& bbox) {}
};

} // namespace isosurface
} // namespace polatory
