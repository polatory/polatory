#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <vector>

namespace polatory::isosurface::snapper {

// A cell of a uniform spatial grid in the isotropic frame, used as a hash-map key (the same
// RowVector3i-plus-hash_combine idiom as rmt's LatticeCoordinates).
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

inline Cell cell_of(const geometry::Point3& p, double cell) {
  return (p / cell).array().floor().cast<int>();
}

// A uniform spatial grid of snap points (in the isotropic frame) with their tolerances, shared by
// the thinner and smoother honor guards to find the points near a local edit. Each point is bucketed
// by the cell it falls in; for_each_near visits every point whose cell meets a query box grown by one
// cell, so a point within one cell-width -- the tolerance bound, since a tolerance is at most the
// resolution -- of the box is never missed.
class PointGrid {
  using Point3 = geometry::Point3;

 public:
  PointGrid(const geometry::Points3& points, const VecX& tolerances, double cell) : cell_(cell) {
    points_.assign(points.rowwise().begin(), points.rowwise().end());
    tol_.resize(points_.size());
    for (std::size_t i = 0; i < points_.size(); i++) {
      tol_.at(i) = tolerances.size() != 0 ? tolerances(static_cast<Index>(i)) : 0.0;
      grid_[cell_of(points_.at(i), cell_)].push_back(static_cast<Index>(i));
    }
  }

  bool empty() const { return points_.empty(); }
  const Point3& point(Index i) const { return points_.at(i); }
  double tolerance(Index i) const { return tol_.at(i); }

  // Invoke fn(i) for each point index in the cells meeting [lo, hi] grown by one cell. fn returns
  // whether to keep going; a false return stops the walk early (for the guards' first failure).
  template <class Fn>
  void for_each_near(const Point3& lo, const Point3& hi, const Fn& fn) const {
    Cell clo = cell_of(lo, cell_);
    Cell chi = cell_of(hi, cell_);
    for (auto i = clo(0) - 1; i <= chi(0) + 1; i++) {
      for (auto j = clo(1) - 1; j <= chi(1) + 1; j++) {
        for (auto k = clo(2) - 1; k <= chi(2) + 1; k++) {
          auto it = grid_.find(Cell(i, j, k));
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

 private:
  std::vector<Point3> points_;
  std::vector<double> tol_;
  double cell_;
  std::unordered_map<Cell, std::vector<Index>, CellHash> grid_;
};

}  // namespace polatory::isosurface::snapper
