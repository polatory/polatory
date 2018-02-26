// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/point_cloud/normal_estimator.hpp>

#include <stdexcept>
#include <tuple>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>

namespace polatory {
namespace point_cloud {

normal_estimator::normal_estimator(const geometry::points3d& points)
  : n_points_(points.rows())
  , points_(points)
  , tree_(points, true) {
}

normal_estimator& normal_estimator::estimate_with_knn(size_t k, double plane_factor_threshold) {
  normals_ = geometry::vectors3d(n_points_, 3);

#pragma omp parallel
  {
    std::vector<size_t> nn_indices;
    std::vector<double> nn_distances;
#pragma omp for
    for (size_t i = 0; i < n_points_; i++) {
      geometry::point3d p = points_.row(i);
      std::tie(nn_indices, nn_distances) = tree_.knn_search(p, k);

      normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
    }
  }

  return *this;
}

normal_estimator& normal_estimator::estimate_with_radius(double radius, double plane_factor_threshold) {
  normals_ = geometry::vectors3d(n_points_, 3);

#pragma omp parallel
  {
    std::vector<size_t> nn_indices;
    std::vector<double> nn_distances;
#pragma omp for
    for (size_t i = 0; i < n_points_; i++) {
      geometry::point3d p = points_.row(i);
      std::tie(nn_indices, nn_distances) = tree_.radius_search(p, radius);

      normals_.row(i) = estimate_impl(nn_indices, plane_factor_threshold);
    }
  }

  return *this;
}

geometry::vectors3d normal_estimator::orient_by_outward_vector(const geometry::vector3d& v) {
  if (n_points_ > 0 && normals_.rows() == 0)
    throw std::runtime_error("Normals have not been estimated yet.");

#pragma omp parallel for
  for (size_t i = 0; i < n_points_; i++) {
    auto n = normals_.row(i);
    if (n.dot(v) < 0.0) {
      n = -n;
    }
  }

  return normals_;
}

geometry::vector3d normal_estimator::estimate_impl(const std::vector<size_t>& nn_indices, double plane_factor_threshold) const {
  if (nn_indices.size() < 3)
    return geometry::vector3d::Zero();

  plane_estimator est(common::take_rows(points_, nn_indices));

  if (est.plane_factor() < plane_factor_threshold)
    return geometry::vector3d::Zero();

  return est.plane_normal();
}

}  // namespace point_cloud
}  // namespace polatory
