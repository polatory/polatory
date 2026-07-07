#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

#include "spatial_grid.hpp"

namespace polatory::isosurface {

class FaceGrid {
  using Point3 = geometry::Point3;

 public:
  FaceGrid(double resolution, Index capacity) : grid_(resolution, capacity), box_(capacity) {}

  // Whether any indexed face whose cells meet the query AABB satisfies `hits`; stops at the first.
  template <class Fn>
  bool any_of(const Point3& lo, const Point3& hi, const Fn& hits) const {
    bool hit = false;
    for_each(lo, hi, [&](Index fi) {
      if (hits(fi)) {
        hit = true;
        return false;
      }
      return true;
    });
    return hit;
  }

  const std::pair<Point3, Point3>& box(Index fi) const { return box_.at(fi); }

  bool empty() const { return grid_.empty(); }

  template <class Fn>
  void for_each(const Point3& lo, const Point3& hi, const Fn& fn) const {
    grid_.for_each(lo, hi, fn);
  }

  void insert(Index fi, const Point3& lo, const Point3& hi) {
    grid_.insert(fi, lo, hi);
    box_.at(fi) = {lo, hi};
  }

  // Inserts a face by the AABB of its vertex positions (the rows of `points`).
  template <class Derived>
  void insert(Index fi, const Eigen::MatrixBase<Derived>& points) {
    insert(fi, Point3(points.colwise().minCoeff()), Point3(points.colwise().maxCoeff()));
  }

  void remove(Index fi) {
    const auto& [lo, hi] = box_.at(fi);
    grid_.remove(fi, lo, hi);
  }

 private:
  SpatialGrid grid_;
  std::vector<std::pair<Point3, Point3>> box_;
};

}  // namespace polatory::isosurface
