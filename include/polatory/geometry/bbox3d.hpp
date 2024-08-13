#pragma once

#include <Eigen/Core>
#include <limits>
#include <polatory/common/io.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

template <int Dim>
class Bbox {
  using Mat = Mat<Dim>;
  using Point = Point<Dim>;
  using Points = Points<Dim>;
  using Vector = Vector<Dim>;

 public:
  Bbox()
      : min_(Point::Constant(std::numeric_limits<double>::infinity())),
        max_(Point::Constant(-std::numeric_limits<double>::infinity())) {}

  Bbox(const Point& min, const Point& max) : min_(min), max_(max) {};

  bool operator==(const Bbox& other) const = default;

  Point center() const { return min_ + width() / 2.0; }

  bool contains(const Point& p) const {
    return (p.array() >= min_.array()).all() && (p.array() <= max_.array()).all();
  }

  Bbox convex_hull(const Bbox& other) const {
    return {min_.cwiseMin(other.min_), max_.cwiseMax(other.max_)};
  }

  Points corners() const {
    Points corners(1 << Dim, Dim);

    for (auto i = 0; i < (1 << Dim); ++i) {
      for (auto j = 0; j < Dim; ++j) {
        corners(i, j) = (i & (1 << j)) != 0 ? max_(j) : min_(j);
      }
    }

    return corners;
  }

  bool is_empty() const { return (min_.array() > max_.array()).any(); }

  const Point& max() const { return max_; }

  const Point& min() const { return min_; }

  template <class Derived>
  static Bbox from_points(const Eigen::MatrixBase<Derived>& points) {
    if (points.rows() == 0) {
      return {};
    }

    return {points.colwise().minCoeff(), points.colwise().maxCoeff()};
  }

  Bbox transform(const Mat& t) const { return from_points(transform_points<Dim>(t, corners())); }

  Vector width() const { return max_ - min_; }

 private:
  POLATORY_FRIEND_READ_WRITE;

  Point min_;
  Point max_;
};

using Bbox1 = Bbox<1>;
using Bbox2 = Bbox<2>;
using Bbox3 = Bbox<3>;

}  // namespace polatory::geometry

namespace polatory::common {

template <int Dim>
struct Read<geometry::Bbox<Dim>> {
  void operator()(std::istream& is, geometry::Bbox<Dim>& t) const {
    read(is, t.min_);
    read(is, t.max_);
  }
};

template <int Dim>
struct Write<geometry::Bbox<Dim>> {
  void operator()(std::ostream& os, const geometry::Bbox<Dim>& t) const {
    write(os, t.min_);
    write(os, t.max_);
  }
};

}  // namespace polatory::common
