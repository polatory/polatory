#include <cmath>
#include <limits>
#include <polatory/common/macros.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>

namespace polatory::point_cloud {

plane_estimator::plane_estimator(const geometry::points3d& points) {
  POLATORY_ASSERT(points.rows() >= 3);

  auto svd = pca_svd(points);

  auto n_points = points.rows();
  auto s0 = svd.singularValues()(0);
  auto s1 = svd.singularValues()(1);
  auto s2 = svd.singularValues()(2);

  point_err_ = std::sqrt(s0 * s0 + s1 * s1 + s2 * s2) / std::sqrt(n_points);
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

double plane_estimator::line_error() const { return line_err_; }

double plane_estimator::plane_factor() const { return plane_factor_; }

geometry::vector3d plane_estimator::plane_normal() const { return basis_.col(2); }

double plane_estimator::plane_error() const { return plane_err_; }

double plane_estimator::point_error() const { return point_err_; }

Eigen::JacobiSVD<Eigen::MatrixXd> plane_estimator::pca_svd(const geometry::points3d& points) {
  geometry::point3d barycenter = points.colwise().mean();
  return Eigen::JacobiSVD<Eigen::MatrixXd>(points.rowwise() - barycenter, Eigen::ComputeThinV);
}

}  // namespace polatory::point_cloud
