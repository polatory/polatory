#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::point_cloud {

class normal_estimator {
 public:
  explicit normal_estimator(const geometry::points3d& points);

  normal_estimator& estimate_with_knn(index_t k, double plane_factor_threshold = 1.8);

  normal_estimator& estimate_with_knn(const std::vector<index_t>& ks,
                                      double plane_factor_threshold = 1.8);

  normal_estimator& estimate_with_radius(double radius, double plane_factor_threshold = 1.8);

  geometry::vectors3d orient_by_outward_vector(const geometry::vector3d& v);

  geometry::vectors3d orient_closed_surface(index_t k);

 private:
  const index_t n_points_;
  const geometry::points3d points_;  // Do not hold a reference to a temporary object.
  kdtree tree_;

  geometry::vectors3d normals_;
};

}  // namespace polatory::point_cloud
