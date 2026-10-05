#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstddef>
#include <jizai/geometry/point3d.hpp>
#include <jizai/types.hpp>
#include <limits>
#include <vector>

namespace jizai::isosurface {

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
  SpatialGrid(double resolution, Index capacity)
      : resolution_(resolution), visited_epoch_(capacity, 0) {}

  bool empty() const { return grid_.empty(); }

  template <class Fn>
  void for_each(const Point3& lo, const Point3& hi, const Fn& fn) const {
    if (epoch_ == std::numeric_limits<int>::max()) {
      std::ranges::fill(visited_epoch_, 0);
      epoch_ = 0;
    }
    epoch_++;

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
            if (visited_epoch_.at(item) == epoch_) {
              continue;
            }
            visited_epoch_.at(item) = epoch_;
            if (!fn(item)) {
              return;
            }
          }
        }
      }
    }
  }

  void insert(Index item, const Point3& lo, const Point3& hi) {
    reserve(item + 1);
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

  void insert_balls(const geometry::Points3& points, const VecX& tols) {
    for (Index i = 0; i < points.rows(); i++) {
      geometry::Vector3 r = geometry::Vector3::Constant(tols(i));
      insert(i, points.row(i) - r, points.row(i) + r);
    }
  }

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

  void reserve(Index capacity) {
    if (static_cast<Index>(visited_epoch_.size()) < capacity) {
      visited_epoch_.resize(capacity, 0);
    }
  }

 private:
  Cell cell_of(const Point3& p) const { return (p / resolution_).array().floor().cast<int>(); }

  mutable int epoch_{};
  boost::unordered_flat_map<Cell, std::vector<Index>, CellHash> grid_;
  double resolution_;
  mutable std::vector<int> visited_epoch_;
};

}  // namespace jizai::isosurface
