// Copyright (c) 2016, GSI and The Polatory Authors.

#include "polatory/point_cloud/normal_estimator.hpp"

#include "polatory/common/vector_view.hpp"
#include "polatory/point_cloud/plane_estimator.hpp"

namespace polatory {
namespace point_cloud {

normal_estimator::normal_estimator(const std::vector<Eigen::Vector3d>& points)
  : points_(points)
  , tree_(points, true) {
}

void normal_estimator::estimate_with_knn(int k, double plane_factor_threshold) {
  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  for (size_t i = 0; i < points_.size(); i++) {
    tree_.knn_search(points_[i], k, nn_indices, nn_distances);

    normals_.push_back(estimate_impl(nn_indices, plane_factor_threshold));
  }
}

void normal_estimator::estimate_with_radius(double radius, double plane_factor_threshold) {
  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  for (size_t i = 0; i < points_.size(); i++) {
    tree_.radius_search(points_[i], radius, nn_indices, nn_distances);

    normals_.push_back(estimate_impl(nn_indices, plane_factor_threshold));
  }
}

const std::vector<Eigen::Vector3d>& normal_estimator::normals() const {
  return normals_;
}

void normal_estimator::orient_normals_by_outward_vector(const Eigen::Vector3d& v) {
  for (auto& n : normals_) {
    if (n.dot(v) < 0.0) {
      n = -n;
    }
  }
}

Eigen::Vector3d normal_estimator::estimate_impl(const std::vector<size_t>& nn_indices, double plane_factor_threshold) const {
  if (nn_indices.size() < 3)
    return Eigen::Vector3d::Zero();

  auto nn_points = common::make_view(points_, nn_indices);
  plane_estimator est(nn_points);

  if (est.plane_factor() < plane_factor_threshold)
    return Eigen::Vector3d::Zero();

  return est.plane_normal();
}

} // namespace point_cloud
} // namespace polatory
