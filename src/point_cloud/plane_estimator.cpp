#include <cmath>
#include <limits>
#include <polatory/common/macros.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>

namespace polatory::point_cloud {

PlaneEstimator::PlaneEstimator(const geometry::Points3& points) {
  POLATORY_ASSERT(points.rows() >= 3);

  auto svd = pca_svd(points);

  auto n_points = points.rows();
  auto s0 = svd.singularValues()(0);
  auto s1 = svd.singularValues()(1);
  auto s2 = svd.singularValues()(2);

  point_err_ = std::hypot(s0, s1, s2) / std::sqrt(n_points);
  line_err_ = std::hypot(s1, s2) / std::sqrt(n_points);
  plane_err_ = std::abs(s2) / std::sqrt(n_points);

  if (s0 == 0.0) {
    plane_factor_ = std::numeric_limits<double>::quiet_NaN();
  } else if (s1 == 0.0) {
    plane_factor_ = 0.0;
  } else if (s2 == 0.0) {
    plane_factor_ = std::numeric_limits<double>::infinity();
  } else {
    plane_factor_ = line_err_ * line_err_ / (plane_err_ * point_err_);
  }

  basis_ = svd.matrixV();
}

double PlaneEstimator::line_error() const { return line_err_; }

double PlaneEstimator::plane_factor() const { return plane_factor_; }

geometry::Vector3 PlaneEstimator::plane_normal() const { return basis_.col(2); }

double PlaneEstimator::plane_error() const { return plane_err_; }

double PlaneEstimator::point_error() const { return point_err_; }

Eigen::JacobiSVD<MatX> PlaneEstimator::pca_svd(const geometry::Points3& points) {
  geometry::Point3 barycenter = points.colwise().mean();
  return Eigen::JacobiSVD<MatX>(points.rowwise() - barycenter, Eigen::ComputeThinV);
}

}  // namespace polatory::point_cloud
