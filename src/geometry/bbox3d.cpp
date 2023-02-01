#include <limits>
#include <polatory/geometry/bbox3d.hpp>

namespace polatory::geometry {

bbox3d::bbox3d()
    : min_(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::infinity()),
      max_(-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity()) {}

bbox3d::bbox3d(const point3d& min, const point3d& max) : min_(min), max_(max) {}

point3d bbox3d::center() const { return min_ + size() / 2.0; }

bool bbox3d::contains(const point3d& p) const {
  return (p.array() >= min_.array()).all() && (p.array() <= max_.array()).all();
}

bbox3d bbox3d::convex_hull(const bbox3d& other) const {
  return {min_.cwiseMin(other.min_), max_.cwiseMax(other.max_)};
}

const point3d& bbox3d::max() const { return max_; }

const point3d& bbox3d::min() const { return min_; }

vector3d bbox3d::size() const { return max_ - min_; }

bbox3d bbox3d::transform(const linear_transformation3d& t) const {
  point3d c = center();
  vector3d v1 = point3d(min_(0), max_(1), max_(2)) - c;
  vector3d v2 = point3d(max_(0), min_(1), max_(2)) - c;
  vector3d v3 = point3d(max_(0), max_(1), min_(2)) - c;

  c = transform_point(t, c);
  v1 = transform_vector(t, v1);
  v2 = transform_vector(t, v2);
  v3 = transform_vector(t, v3);

  vectors3d vertices(8, 3);
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

  return {min, max};
}

bbox3d bbox3d::from_points(const points3d& points) {
  return from_points(points.rowwise().begin(), points.rowwise().end());
}

}  // namespace polatory::geometry
