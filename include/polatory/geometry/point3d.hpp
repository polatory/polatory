// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace polatory {
namespace geometry {

using vector3d = Eigen::RowVector3d;

using point3d = vector3d;

class vectors3d : public Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> {
  using base_type = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

public:
  using base_type::base_type;

  vectors3d()
    : base_type() {
  }

  explicit vectors3d(int n)
    : base_type(n, 3) {
  }

  explicit vectors3d(unsigned int n)
    : base_type(n, 3) {
  }

  explicit vectors3d(ptrdiff_t n)
    : base_type(n, 3) {
  }

  explicit vectors3d(size_t n)
    : base_type(n, 3) {
  }
};

using points3d = vectors3d;

} // namespace geometry
} // namespace polatory
