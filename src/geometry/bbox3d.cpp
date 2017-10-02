// Copyright (c) 2016, GSI and The Polatory Authors.

#include "polatory/geometry/bbox3d.hpp"

#include <limits>

namespace polatory {
namespace geometry {

bbox3d::bbox3d()
  : min_(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())
  , max_(-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()) {
}

bbox3d::bbox3d(const Eigen::Vector3d& min, const Eigen::Vector3d& max)
  : min_(min)
  , max_(max) {
}

Eigen::Vector3d bbox3d::center() const {
  return (min_ + max_) / 2.0;
}

const Eigen::Vector3d& bbox3d::max() const {
  return max_;
}

const Eigen::Vector3d& bbox3d::min() const {
  return min_;
}

Eigen::Vector3d bbox3d::size() const {
  return max_ - min_;
}

bbox3d bbox3d::transform(const affine_transform3d& affine) const {
  Eigen::Vector3d c = center();
  Eigen::Vector3d v1 = Eigen::Vector3d(min_(0), max_(1), max_(2)) - c;
  Eigen::Vector3d v2 = Eigen::Vector3d(max_(0), min_(1), max_(2)) - c;
  Eigen::Vector3d v3 = Eigen::Vector3d(max_(0), max_(1), min_(2)) - c;

  c = affine.transform_point(c);
  v1 = affine.transform_vector(v1);
  v2 = affine.transform_vector(v2);
  v3 = affine.transform_vector(v3);

  Eigen::MatrixXd vertices(3, 8);
  vertices.col(0) = -v1 - v2 - v3;  // min, min, min
  vertices.col(1) = -v1;            // max, min, min
  vertices.col(2) = v3;             // max, max, min
  vertices.col(3) = -v2;            // min, max, min
  vertices.col(4) = v1;             // min, max, max
  vertices.col(5) = -v3;            // min, min, max
  vertices.col(6) = v2;             // max, min, max
  vertices.col(7) = v1 + v2 + v3;   // max, max, max

  Eigen::Vector3d min = c + Eigen::Vector3d(
    vertices.row(0).minCoeff(),
    vertices.row(1).minCoeff(),
    vertices.row(2).minCoeff()
  );
  Eigen::Vector3d max = c + Eigen::Vector3d(
    vertices.row(0).maxCoeff(),
    vertices.row(1).maxCoeff(),
    vertices.row(2).maxCoeff()
  );

  return bbox3d(min, max);
}

} // namespace geometry
} // namespace polatory
