// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <iterator>

#include <Eigen/Core>

namespace polatory {
namespace geometry {

class cuboid3d {
public:
  cuboid3d()
    : min_(Eigen::Vector3d::Zero())
    , max_(Eigen::Vector3d::Ones()) {
  }

  cuboid3d(const Eigen::Vector3d& min, const Eigen::Vector3d& max)
    : min_(min)
    , max_(max) {
  }

  const Eigen::Vector3d& max() const {
    return max_;
  }

  const Eigen::Vector3d& min() const {
    return min_;
  }

private:
  const Eigen::Vector3d min_;
  const Eigen::Vector3d max_;
};

} // namespace geometry
} // namespace polatory
