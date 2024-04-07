#pragma once

#include <Eigen/SVD>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::point_cloud {

// Computes the best-fit plane and its "plane factor" for the given points.
class plane_estimator {
 public:
  explicit plane_estimator(const geometry::points3d& points);

  double line_error() const;

  double plane_factor() const;

  geometry::vector3d plane_normal() const;

  double plane_error() const;

  double point_error() const;

 private:
  static Eigen::JacobiSVD<matrixd> pca_svd(const geometry::points3d& points);

  geometry::matrix3d basis_;

  double point_err_;
  double line_err_;
  double plane_err_;
  double plane_factor_;
};

}  // namespace polatory::point_cloud
