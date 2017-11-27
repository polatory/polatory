// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/point_cloud/normal_estimator.hpp>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>

namespace polatory {
namespace point_cloud {

normal_estimator::normal_estimator(const geometry::points3d& points)
  : n_points(points.rows())
  , points_(points)
  , tree_(points, true) {
}

void normal_estimator::estimate_with_knn(int k, double plane_factor_threshold) {
  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  normals_ = geometry::vectors3d(n_points);
  for (size_t i = 0; i < n_points; i++) {
    tree_.knn_search(points_.row(i), k, nn_indices, nn_distances);

    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
  }
}

void normal_estimator::estimate_with_radius(double radius, double plane_factor_threshold) {
  std::vector<size_t> nn_indices;
  std::vector<double> nn_distances;

  normals_ = geometry::vectors3d(n_points);
  for (size_t i = 0; i < n_points; i++) {
    tree_.radius_search(points_.row(i), radius, nn_indices, nn_distances);

    normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
  }
}

const geometry::vectors3d& normal_estimator::normals() const {
  return normals_;
}

void normal_estimator::orient_normals_by_outward_vector(const geometry::vector3d& v) {
  for (auto n : common::row_range(normals_)) {
    if (n.dot(v) < 0.0) {
      n = -n;
    }
  }
}

geometry::vector3d normal_estimator::estimate_impl(const std::vector<size_t>& nn_indices, double plane_factor_threshold) const {
  if (nn_indices.size() < 3)
    return geometry::vector3d::Zero();

  plane_estimator est(common::take_rows(points_, nn_indices));

  if (est.plane_factor() < plane_factor_threshold)
    return geometry::vector3d::Zero();

  return est.plane_normal();
}

} // namespace point_cloud
} // namespace polatory
