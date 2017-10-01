// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace isosurface {

struct field_function {
  virtual ~field_function() {}

  virtual Eigen::VectorXd operator()(const std::vector<Eigen::Vector3d>& points) const = 0;
};

} // namespace isosurface
} // namespace polatory
