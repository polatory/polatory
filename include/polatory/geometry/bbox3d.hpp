#pragma once

#include <limits>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

template <int Dim>
class bboxNd {
  using Point = pointNd<Dim>;
  using Vector = vectorNd<Dim>;
  using Matrix = matrixNd<Dim>;

 public:
  bboxNd()
      : min_(Point::Constant(std::numeric_limits<double>::infinity())),
        max_(Point::Constant(-std::numeric_limits<double>::infinity())) {}

  bboxNd(const Point& min, const Point& max) : min_(min), max_(max){};

  bool operator==(const bboxNd& other) const = default;

  Point center() const { return min_ + size() / 2.0; }

  bool contains(const Point& p) const {
    return (p.array() >= min_.array()).all() && (p.array() <= max_.array()).all();
  }

  bboxNd convex_hull(const bboxNd& other) const {
    return {min_.cwiseMin(other.min_), max_.cwiseMax(other.max_)};
  }

  const Point& max() const { return max_; }

  const Point& min() const { return min_; }

  Vector size() const { return max_ - min_; }

  bboxNd transform(const Matrix& t) const {
    geometry::pointsNd<Dim> corners(1 << Dim, Dim);

    for (auto i = 0; i < (1 << Dim); ++i) {
      for (auto j = 0; j < Dim; ++j) {
        corners(i, j) = (i & (1 << j)) ? max_(j) : min_(j);
      }
      corners.row(i) = transform_point<Dim>(t, corners.row(i));
    }

    return from_points(corners);
  }

  template <class Derived>
  static bboxNd from_points(const Eigen::MatrixBase<Derived>& points) {
    if (points.rows() == 0) {
      return {};
    }

    return {points.colwise().minCoeff(), points.colwise().maxCoeff()};
  }

 private:
  Point min_;
  Point max_;
};

using bbox3d = bboxNd<3>;

}  // namespace polatory::geometry
