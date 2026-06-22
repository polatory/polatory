#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <vector>

namespace polatory::isosurface::snapper {

using Cell = Eigen::RowVector3i;

struct CellHash {
  std::size_t operator()(const Cell& c) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, c(0));
    boost::hash_combine(seed, c(1));
    boost::hash_combine(seed, c(2));
    return seed;
  }
};

inline Cell cell_of(const geometry::Point3& p, double resolution) {
  return (p / resolution).array().floor().cast<int>();
}

class PointGrid {
  using Point3 = geometry::Point3;

 public:
  PointGrid(const geometry::Points3& points, const VecX& tolerances, double resolution)
      : points_(points.rowwise().begin(), points.rowwise().end()),
        tol_(points_.size()),
        resolution_(resolution) {
    for (std::size_t i = 0; i < points_.size(); i++) {
      tol_.at(i) = tolerances.size() != 0 ? tolerances(static_cast<Index>(i)) : 0.0;
      grid_[cell_of(points_.at(i), resolution_)].push_back(static_cast<Index>(i));
    }
  }

  bool empty() const { return points_.empty(); }

  // The one-cell margin is enough because a tolerance is at most one cell (the resolution).
  template <class Fn>
  void for_each_near(const Point3& lo, const Point3& hi, const Fn& fn) const {
    auto clo = cell_of(lo, resolution_);
    auto chi = cell_of(hi, resolution_);
    for (auto i = clo(0) - 1; i <= chi(0) + 1; i++) {
      for (auto j = clo(1) - 1; j <= chi(1) + 1; j++) {
        for (auto k = clo(2) - 1; k <= chi(2) + 1; k++) {
          auto it = grid_.find(Cell{i, j, k});
          if (it == grid_.end()) {
            continue;
          }
          for (Index pi : it->second) {
            if (!fn(pi)) {
              return;
            }
          }
        }
      }
    }
  }

  const Point3& point(Index i) const { return points_.at(i); }

  double tolerance(Index i) const { return tol_.at(i); }

 private:
  std::vector<Point3> points_;
  std::vector<double> tol_;
  double resolution_;
  std::unordered_map<Cell, std::vector<Index>, CellHash> grid_;
};

}  // namespace polatory::isosurface::snapper
