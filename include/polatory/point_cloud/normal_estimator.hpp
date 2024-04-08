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

  normal_estimator& estimate_with_knn(index_t k);

  normal_estimator& estimate_with_knn(const std::vector<index_t>& ks);

  normal_estimator& estimate_with_radius(double radius);

  normal_estimator& estimate_with_radius(const std::vector<double>& radii);

  normal_estimator& filter_by_plane_factor(double threshold = 1.8);

  const geometry::vectors3d& normals() const {
    throw_if_not_estimated();

    return normals_;
  }

  normal_estimator& orient_toward_direction(const geometry::vector3d& direction);

  normal_estimator& orient_toward_point(const geometry::point3d& point);

  normal_estimator& orient_closed_surface(index_t k);

  const vectord& plane_factors() const {
    throw_if_not_estimated();

    return plane_factors_;
  }

 private:
  void throw_if_not_estimated() const {
    if (!estimated_) {
      throw std::runtime_error("normals have not been estimated");
    }
  }

  const index_t n_points_;
  const geometry::points3d points_;  // Do not hold a reference to a temporary object.
  kdtree<3> tree_;

  bool estimated_{};
  geometry::vectors3d normals_;
  vectord plane_factors_;
};

}  // namespace polatory::point_cloud
