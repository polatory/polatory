#pragma once

#include <Eigen/Core>
#include <jizai/geometry/point3d.hpp>
#include <jizai/point_cloud/kdtree.hpp>
#include <jizai/point_cloud/plane_estimator.hpp>
#include <jizai/types.hpp>
#include <vector>

namespace jizai::point_cloud {

class NormalEstimator {
 public:
  explicit NormalEstimator(const geometry::Points3& points);

  void estimate_with_knn(Index k);

  void estimate_with_knn(const std::vector<Index>& ks);

  void estimate_with_radius(double radius);

  void estimate_with_radius(const std::vector<double>& radii);

  void filter_by_plane_factor(double threshold = 1.8);

  const geometry::Vectors3& normals() const {
    throw_if_not_estimated();

    return normals_;
  }

  void orient_toward_direction(const geometry::Vector3& direction);

  void orient_toward_point(const geometry::Point3& point);

  void orient_closed_surface(Index k = 100);

  const VecX& plane_factors() const {
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

}  // namespace jizai::point_cloud
