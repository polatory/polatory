#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <boost/unordered/unordered_flat_set.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/point_cloud/kdtree.hpp>
#include <jizai/types.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace jizai::point_cloud {

template <int Dim>
class DistanceFilter {
  using Points = geometry::Points<Dim>;

 public:
  explicit DistanceFilter(const Points& points) : points_(points), tree_(points_) {}

  std::vector<Index> filtered_indices(double distance = 0.0) const {
    return filtered_indices(distance, trivial_indices(points_.rows()));
  }

  std::vector<Index> filtered_indices(const std::vector<Index>& indices) const {
    return filtered_indices(0.0, indices);
  }

  std::vector<Index> filtered_indices(double distance, const std::vector<Index>& indices) const {
    if (!(distance >= 0.0)) {
      throw std::invalid_argument("distance must be non-negative");
    }

    if (!std::ranges::all_of(indices, [&](auto i) { return i >= 0 && i < points_.rows(); })) {
      throw std::invalid_argument("indices must be in [0, points.rows())");
    }

    boost::unordered_flat_set<Index> indices_to_remove;

    std::vector<Index> nn_indices;
    std::vector<double> nn_distances;
    for (auto i : indices) {
      if (indices_to_remove.contains(i)) {
        continue;
      }

      auto p = points_.row(i);
      tree_.radius_search(p, distance, nn_indices, nn_distances);

      for (auto j : nn_indices) {
        if (j != i) {
          indices_to_remove.insert(j);
        }
      }
    }

    std::vector<Index> filtered_indices;
    for (auto i : indices) {
      if (!indices_to_remove.contains(i)) {
        filtered_indices.push_back(i);
      }
    }

    return filtered_indices;
  }

 private:
  static std::vector<Index> trivial_indices(Index n_points) {
    std::vector<Index> indices(n_points);
    std::iota(indices.begin(), indices.end(), Index{0});
    return indices;
  }

  const Points points_;  // Do not hold a reference to a temporary object.
  const KdTree<Dim> tree_;
};

}  // namespace jizai::point_cloud
