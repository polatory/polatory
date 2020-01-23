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

using linear_transformation3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

template <class T>
linear_transformation3d to_linear_transformation3d(T t) {
  return Eigen::Transform<double, 3, Eigen::Affine, Eigen::RowMajor>(t).linear();
}

}  // namespace geometry
}  // namespace polatory
