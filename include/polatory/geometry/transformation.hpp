// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace geometry {

Eigen::Matrix3d scaling(const vector3d& scales);

Eigen::Matrix3d rotation(double angle, int axis);

}  // namespace geometry
}  // namespace polatory
