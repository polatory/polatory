#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <vector>

namespace polatory::isosurface::snapper {

// A uniform spatial grid mapping cells to item indices. An item is inserted over every cell its
// world AABB touches (a point is a degenerate AABB), and for_each visits each distinct item whose
// AABB shares a cell with the query AABB. Geometry stays with the caller; the grid holds only
// indices, so the same grid serves snap points (a point as a tolerance-radius ball) and faces (a
// triangle as its AABB) alike.
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
  // capacity is the number of items; their indices index the per-query dedup stamp.
  SpatialGrid(double resolution, Index capacity)
      : resolution_(resolution), visited_(capacity, -1) {}

  bool empty() const { return grid_.empty(); }

  // Visits each distinct item whose cells meet the query AABB, until fn returns false.
  template <class Fn>
  void for_each(const Point3& lo, const Point3& hi, const Fn& fn) const {
    guard_++;
    auto clo = cell_of(lo);
    auto chi = cell_of(hi);
    for (auto i = clo(0); i <= chi(0); i++) {
      for (auto j = clo(1); j <= chi(1); j++) {
        for (auto k = clo(2); k <= chi(2); k++) {
          auto it = grid_.find(Cell{i, j, k});
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
          grid_[Cell{i, j, k}].push_back(item);
        }
      }
    }
  }

  void insert(Index item, const Point3& p) { insert(item, p, p); }

 private:
  Cell cell_of(const Point3& p) const { return (p / resolution_).array().floor().cast<int>(); }

  double resolution_;
  std::unordered_map<Cell, std::vector<Index>, CellHash> grid_;
  mutable std::vector<Index> visited_;  // per-query dedup stamp
  mutable Index guard_{};
};

}  // namespace polatory::isosurface::snapper
