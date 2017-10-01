// Copyright (c) 2016, GSI and The Polatory Authors.

#include "point_cloud/distance_filter.hpp"

#include <cassert>
#include <set>
#include <vector>

#include <Eigen/Core>

#include "common/vector_view.hpp"
#include "point_cloud/kdtree.hpp"

namespace polatory {
namespace point_cloud {

distance_filter::distance_filter(const std::vector<Eigen::Vector3d>& points, double distance)
  : points_(points)
  , n_points_(points.size()) {
  kdtree tree(points);
  tree.set_exact_search();

  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  std::set<size_t> indices_to_remove;

  for (size_t i = 0; i < n_points_; i++) {
    if (indices_to_remove.find(i) != indices_to_remove.end())
      continue;

    auto found = tree.radius_search(points[i], distance, nn_indices, nn_distances);

    for (int k = 0; k < found; k++) {
      auto j = nn_indices[k];

      if (j != i) {
        indices_to_remove.insert(j);
      }
    }
  }

  for (size_t i = 0; i < n_points_; i++) {
    if (indices_to_remove.count(i) == 0) {
      filtered_indices_.push_back(i);
    }
  }
}

const std::vector<size_t>& distance_filter::filtered_indices() const {
  return filtered_indices_;
}

std::vector<Eigen::Vector3d> distance_filter::filtered_points() const {
  auto view = common::make_view(points_, filtered_indices_);

  return std::vector<Eigen::Vector3d>(view.begin(), view.end());
}

Eigen::VectorXd distance_filter::filter_values(const Eigen::VectorXd& values) const {
  assert(values.size() == n_points_);

  Eigen::VectorXd filtered(filtered_indices_.size());

  for (size_t i = 0; i < filtered_indices_.size(); i++) {
    filtered(i) = values(filtered_indices_[i]);
  }

  return filtered;
}

} // namespace point_cloud
} // namespace polatory
