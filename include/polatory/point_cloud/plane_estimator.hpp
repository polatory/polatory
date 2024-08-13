#pragma once

#include <Eigen/SVD>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::point_cloud {

// Computes the best-fit plane and its "plane factor" for the given points.
class PlaneEstimator {
 public:
  explicit PlaneEstimator(const geometry::Points3& points);

  double line_error() const;

  double plane_factor() const;

  geometry::Vector3 plane_normal() const;

  double plane_error() const;

  double point_error() const;

 private:
  static Eigen::JacobiSVD<MatX> pca_svd(const geometry::Points3& points);

  Mat3 basis_;

  double point_err_;
  double line_err_;
  double plane_err_;
  double plane_factor_;
};

}  // namespace polatory::point_cloud
