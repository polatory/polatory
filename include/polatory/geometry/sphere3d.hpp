// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <iterator>

#include <Eigen/Core>

namespace polatory {
namespace geometry {

class sphere3d {
public:
  sphere3d()
    : center_(Eigen::Vector3d::Zero())
    , radius_(1.0) {
  }

  sphere3d(const Eigen::Vector3d& center, double radius)
    : center_(center)
    , radius_(radius) {
  }

  const Eigen::Vector3d& center() const {
    return center_;
  }

  double radius() const {
    return radius_;
  }

private:
  const Eigen::Vector3d center_;
  const double radius_;
};

} // namespace geometry
} // namespace polatory
