// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/geometry/bbox3d.hpp>

#include <limits>

#include <polatory/common/eigen_utility.hpp>

namespace polatory {
namespace geometry {

bbox3d::bbox3d()
  : min_(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())
  , max_(-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()) {
}

bbox3d::bbox3d(const point3d& min, const point3d& max)
  : min_(min)
  , max_(max) {
}

bool bbox3d::operator==(const bbox3d& other) const {
  return min_ == other.min_ && max_ == other.max_;
}

point3d bbox3d::center() const {
  return min_ + size() / 2.0;
}

const point3d& bbox3d::max() const {
  return max_;
}

const point3d& bbox3d::min() const {
  return min_;
}

vector3d bbox3d::size() const {
  return max_ - min_;
}

bbox3d bbox3d::transform(const affine_transform3d& affine) const {
  point3d c = center();
  vector3d v1 = point3d(min_(0), max_(1), max_(2)) - c;
  vector3d v2 = point3d(max_(0), min_(1), max_(2)) - c;
  vector3d v3 = point3d(max_(0), max_(1), min_(2)) - c;

  c = affine.transform_point(c);
  v1 = affine.transform_vector(v1);
  v2 = affine.transform_vector(v2);
  v3 = affine.transform_vector(v3);

  points3d vertices(8);
  vertices.row(0) = -v1 - v2 - v3;  // min, min, min
  vertices.row(1) = -v1;            // max, min, min
  vertices.row(2) = v3;             // max, max, min
  vertices.row(3) = -v2;            // min, max, min
  vertices.row(4) = v1;             // min, max, max
  vertices.row(5) = -v3;            // min, min, max
  vertices.row(6) = v2;             // max, min, max
  vertices.row(7) = v1 + v2 + v3;   // max, max, max

  point3d min = c + vertices.colwise().minCoeff();
  point3d max = c + vertices.colwise().maxCoeff();

  return bbox3d(min, max);
}

bbox3d bbox3d::union_hull(const bbox3d& other) const {
  return bbox3d(
    min().cwiseMin(other.min()),
    max().cwiseMax(other.max())
  );
}

bbox3d bbox3d::from_points(const points3d& points) {
  return from_points(common::row_begin(points), common::row_end(points));
}

} // namespace geometry
} // namespace polatory
