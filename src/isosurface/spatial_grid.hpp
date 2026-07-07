#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstddef>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

// A uniform grid mapping cells to item indices: an item is inserted over the cells its world AABB
// touches, and for_each visits each distinct item near a query AABB. Holds only indices (geometry
// stays with the caller), so it serves both snap points (as tolerance-radius balls) and faces.
class SpatialGrid {
  using Point3 = geometry::Point3;
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

 public:
  // capacity = item count (sizes the dedup stamp).
  SpatialGrid(double resolution, Index capacity) : resolution_(resolution), visited_(capacity, 0) {}

  bool empty() const { return grid_.empty(); }

  // Visits each distinct item whose cells meet the query AABB, until fn returns false.
  template <class Fn>
  void for_each(const Point3& lo, const Point3& hi, const Fn& fn) const {
    // Reset before signed overflow (UB); at one bump per query this fires astronomically rarely.
    if (guard_ == std::numeric_limits<int>::max()) {
      std::ranges::fill(visited_, 0);
      guard_ = 0;
    }
    guard_++;

    auto clo = cell_of(lo);
    auto chi = cell_of(hi);
    for (auto i = clo(0); i <= chi(0); i++) {
      for (auto j = clo(1); j <= chi(1); j++) {
        for (auto k = clo(2); k <= chi(2); k++) {
          auto it = grid_.find({i, j, k});
          if (it == grid_.end()) {
            continue;
          }
          for (Index item : it->second) {
            if (visited_.at(item) == guard_) {
              continue;
            }
            visited_.at(item) = guard_;
            if (!fn(item)) {
              return;
            }
          }
        }
      }
    }
  }

  void insert(Index item, const Point3& lo, const Point3& hi) {
    auto clo = cell_of(lo);
    auto chi = cell_of(hi);
    for (auto i = clo(0); i <= chi(0); i++) {
      for (auto j = clo(1); j <= chi(1); j++) {
        for (auto k = clo(2); k <= chi(2); k++) {
          grid_[{i, j, k}].push_back(item);
        }
      }
    }
  }

  void insert(Index item, const Point3& p) { insert(item, p, p); }

  // Insert each point as a tolerance-radius ball, so a query AABB finds every point it reaches.
  void insert_balls(const geometry::Points3& points, const VecX& tols) {
    for (Index i = 0; i < points.rows(); i++) {
      geometry::Vector3 r = geometry::Vector3::Constant(tols(i));
      insert(i, points.row(i) - r, points.row(i) + r);
    }
  }

  // Pass the same AABB the item was inserted with, else stale cell entries are left behind.
  void remove(Index item, const Point3& lo, const Point3& hi) {
    auto clo = cell_of(lo);
    auto chi = cell_of(hi);
    for (auto i = clo(0); i <= chi(0); i++) {
      for (auto j = clo(1); j <= chi(1); j++) {
        for (auto k = clo(2); k <= chi(2); k++) {
          auto it = grid_.find({i, j, k});
          if (it != grid_.end()) {
            std::erase(it->second, item);
          }
        }
      }
    }
  }

 private:
  Cell cell_of(const Point3& p) const { return (p / resolution_).array().floor().cast<int>(); }

  double resolution_;
  boost::unordered_flat_map<Cell, std::vector<Index>, CellHash> grid_;
  mutable std::vector<int> visited_;  // per-query dedup stamp (epoch of last visit)
  mutable int guard_{};
};

}  // namespace polatory::isosurface
