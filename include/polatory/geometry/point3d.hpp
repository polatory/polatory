// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace polatory {
namespace geometry {

using vector3d = Eigen::RowVector3d;

using point3d = vector3d;

using vectors3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

using points3d = vectors3d;

}  // namespace geometry
}  // namespace polatory
