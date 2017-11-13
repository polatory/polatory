// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/point_cloud/distance_filter.hpp>

#include <cassert>
#include <set>

#include <polatory/common/exception.hpp>
#include <polatory/common/vector_view.hpp>
#include <polatory/point_cloud/kdtree.hpp>

namespace polatory {
namespace point_cloud {

distance_filter::distance_filter(const std::vector<Eigen::Vector3d>& points, double distance)
  : n_points_(points.size()) {
  if (distance <= 0.0)
    throw common::invalid_argument("distance > 0.0");

  kdtree tree(points, true);

  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  std::set<size_t> indices_to_remove;

  for (size_t i = 0; i < n_points_; i++) {
    if (indices_to_remove.find(i) != indices_to_remove.end())
      continue;

    tree.radius_search(points[i], distance, nn_indices, nn_distances);

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

const std::vector<size_t>& distance_filter::filtered_indices() const {
  return filtered_indices_;
}

std::vector<Eigen::Vector3d> distance_filter::filter_points(const std::vector<Eigen::Vector3d>& points) const {
  assert(points.size() == n_points_);

  std::vector<Eigen::Vector3d> filtered;
  filtered.reserve(filtered_indices_.size());

  for (auto idx : filtered_indices_) {
    filtered.push_back(points[idx]);
  }

  return filtered;
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
