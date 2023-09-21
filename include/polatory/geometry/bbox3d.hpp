#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory::geometry {

class bbox3d {
 public:
  bbox3d();

  bbox3d(const point3d& min, const point3d& max);

  bool operator==(const bbox3d& other) const = default;

  point3d center() const;

  bool contains(const point3d& p) const;

  bbox3d convex_hull(const bbox3d& other) const;

  const point3d& max() const;

  const point3d& min() const;

  vector3d size() const;

  bbox3d transform(const linear_transformation3d& t) const;

  template <class Derived>
  static bbox3d from_points(const Eigen::MatrixBase<Derived>& points) {
    if (points.rows() == 0) {
      return {};
    }

    return {points.colwise().minCoeff(), points.colwise().maxCoeff()};
  }

 private:
  point3d min_;
  point3d max_;
};

}  // namespace polatory::geometry
