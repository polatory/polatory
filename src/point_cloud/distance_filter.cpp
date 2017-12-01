// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/point_cloud/distance_filter.hpp>

#include <set>

#include <polatory/point_cloud/kdtree.hpp>

namespace polatory {
namespace point_cloud {

distance_filter::distance_filter(const geometry::points3d& points, double distance)
  : n_points_(points.rows()) {
  if (distance <= 0.0)
    throw common::invalid_argument("distance > 0.0");

  kdtree tree(points, true);

  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  std::set<size_t> indices_to_remove;

  for (size_t i = 0; i < n_points_; i++) {
    if (indices_to_remove.find(i) != indices_to_remove.end())
      continue;

    tree.radius_search(points.row(i), distance, nn_indices, nn_distances);

    for (auto j : nn_indices) {
      if (j != i) {
        indices_to_remove.insert(j);
      }
    }
  }

  for (size_t i = 0; i < n_points_; i++) {
    if (indices_to_remove.find(i) == indices_to_remove.end()) {
      filtered_indices_.push_back(i);
    }
  }
}

} // namespace point_cloud
} // namespace polatory
