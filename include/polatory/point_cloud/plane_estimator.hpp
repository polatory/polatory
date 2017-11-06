// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace polatory {
namespace point_cloud {

// Computes the best-fit plane and its "plane factor" for the given points.
class plane_estimator {
public:
  template <class Container>
  explicit plane_estimator(const Container& points)
    : center_(barycenter(points)) {
    assert(points.size() >= 3);

    auto svd = pca_svd(points);

    auto n_points = points.size();
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

  double line_error() const {
    return line_err_;
  }

  double plane_factor() const {
    return plane_factor_;
  }

  Eigen::Vector3d plane_normal() const {
    return basis_.col(2);
  }

  double plane_error() const {
    return plane_err_;
  }

  double point_error() const {
    return point_err_;
  }

private:
  template <class Container>
  Eigen::Vector3d barycenter(const Container& points) const {
    return std::accumulate(
      points.begin(), points.end(), Eigen::Vector3d(Eigen::Vector3d::Zero())
    ) / points.size();
  }

  template <class Container>
  Eigen::JacobiSVD<Eigen::MatrixXd> pca_svd(const Container& points) const {
    Eigen::MatrixXd mat(points.size(), 3);
    for (size_t i = 0; i < points.size(); i++) {
      mat.row(i) = points[i] - center_;
    }

    return Eigen::JacobiSVD<Eigen::MatrixXd>(mat, Eigen::ComputeThinV);
  }

  Eigen::Vector3d center_;
  Eigen::Matrix3d basis_;

  double point_err_;
  double line_err_;
  double plane_err_;
  double plane_factor_;
};

} // namespace point_cloud
} // namespace polatory
