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

  point_err_ = std::max(point_err_, 1e-10);
  line_err_ = std::max(line_err_, 1e-10 * point_err_);
  plane_err_ = std::max(plane_err_, 1e-10 * line_err_);

  plane_factor_ = line_err_ * line_err_ / (plane_err_ * point_err_);

  basis_ = svd.matrixV();
}

double PlaneEstimator::line_error() const { return line_err_; }

double PlaneEstimator::plane_factor() const { return plane_factor_; }

geometry::Vector3 PlaneEstimator::plane_normal() const { return basis_.col(2); }

double PlaneEstimator::plane_error() const { return plane_err_; }

double PlaneEstimator::point_error() const { return point_err_; }

Eigen::JacobiSVD<geometry::Points3> PlaneEstimator::pca_svd(const geometry::Points3& points) {
  geometry::Point3 barycenter = points.colwise().mean();
  return Eigen::JacobiSVD<geometry::Points3>(points.rowwise() - barycenter, Eigen::ComputeFullV);
}

}  // namespace polatory::point_cloud
