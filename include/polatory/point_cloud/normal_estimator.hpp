// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>

namespace polatory {
namespace point_cloud {

class normal_estimator {
public:
  normal_estimator(const geometry::points3d& points);

  void estimate_with_knn(int k, double plane_factor_threshold = 1.8);

  void estimate_with_radius(double radius, double plane_factor_threshold = 1.8);

  const geometry::vectors3d& normals() const;

  void orient_normals_by_outward_vector(const geometry::vector3d& v);

private:
  geometry::vector3d estimate_impl(const std::vector<size_t>& nn_indices, double plane_factor_threshold) const;

  const size_t n_points;
  const geometry::points3d& points_;
  kdtree tree_;

  geometry::vectors3d normals_;
};

} // namespace point_cloud
} // namespace polatory
