#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::point_cloud {

class NormalEstimator {
 public:
  explicit NormalEstimator(const geometry::Points3& points);

  NormalEstimator& estimate_with_knn(Index k) &;

  NormalEstimator&& estimate_with_knn(Index k) && { return std::move(estimate_with_knn(k)); }

  NormalEstimator& estimate_with_knn(const std::vector<Index>& ks) &;

  NormalEstimator&& estimate_with_knn(const std::vector<Index>& ks) && {
    return std::move(estimate_with_knn(ks));
  }

  NormalEstimator& estimate_with_radius(double radius) &;

  NormalEstimator&& estimate_with_radius(double radius) && {
    return std::move(estimate_with_radius(radius));
  }

  NormalEstimator& estimate_with_radius(const std::vector<double>& radii) &;

  NormalEstimator&& estimate_with_radius(const std::vector<double>& radii) && {
    return std::move(estimate_with_radius(radii));
  }

  NormalEstimator& filter_by_plane_factor(double threshold = 1.8) &;

  NormalEstimator&& filter_by_plane_factor(double threshold = 1.8) && {
    return std::move(filter_by_plane_factor(threshold));
  }

  geometry::Vectors3&& into_normals() && {
    throw_if_not_estimated();

    return std::move(normals_);
  }

  const geometry::Vectors3& normals() const& {
    throw_if_not_estimated();

    return normals_;
  }

  NormalEstimator& orient_toward_direction(const geometry::Vector3& direction) &;

  NormalEstimator&& orient_toward_direction(const geometry::Vector3& direction) && {
    return std::move(orient_toward_direction(direction));
  }

  NormalEstimator& orient_toward_point(const geometry::Point3& point) &;

  NormalEstimator&& orient_toward_point(const geometry::Point3& point) && {
    return std::move(orient_toward_point(point));
  }

  NormalEstimator& orient_closed_surface(Index k = 100) &;

  NormalEstimator&& orient_closed_surface(Index k = 100) && {
    return std::move(orient_closed_surface(k));
  }

  const VecX& plane_factors() const& {
    throw_if_not_estimated();

    return plane_factors_;
  }

 private:
  void throw_if_not_estimated() const {
    if (!estimated_) {
      throw std::runtime_error("normals have not been estimated");
    }
  }

  const Index n_points_;
  const geometry::Points3 points_;  // Do not hold a reference to a temporary object.
  KdTree<3> tree_;

  bool estimated_{};
  geometry::Vectors3 normals_;
  VecX plane_factors_;
};

}  // namespace polatory::point_cloud
