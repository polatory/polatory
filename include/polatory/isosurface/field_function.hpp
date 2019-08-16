// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace isosurface {

struct field_function {
  virtual ~field_function() = default;

  virtual common::valuesd operator()(const geometry::points3d& points) const = 0;

  virtual void set_evaluation_bbox(const geometry::bbox3d& /*bbox*/) {}
};

}  // namespace isosurface
}  // namespace polatory
