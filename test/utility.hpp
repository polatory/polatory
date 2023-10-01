#pragma once

#include <Eigen/Core>
#include <cmath>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <utility>

template <int Dim>
polatory::geometry::matrixNd<Dim> random_anisotropy() {
  using Matrix = polatory::geometry::matrixNd<Dim>;

  Matrix a = Matrix::Random();
  if (a.determinant() < 0.0) {
    a.col(0) *= -1.0;
  }

  return a;
}

template <class DerivedApprox, class DerivedExact>
double relative_error(const Eigen::MatrixBase<DerivedApprox>& approx,
                      const Eigen::MatrixBase<DerivedExact>& exact) {
  auto error = (approx - exact).template lpNorm<Eigen::Infinity>();
  auto scale = exact.template lpNorm<Eigen::Infinity>();
  return error / scale;
}

template <int Dim>
std::pair<polatory::geometry::pointsNd<Dim>, polatory::common::valuesd> sample_data(
    polatory::index_t n_points, const polatory::geometry::matrixNd<Dim>& aniso) {
  using polatory::index_t;
  using polatory::common::valuesd;
  using polatory::point_cloud::distance_filter;
  using Point = polatory::geometry::pointNd<Dim>;
  using Points = polatory::geometry::pointsNd<Dim>;

  Points points = Points::Random(n_points, Dim);
  points = distance_filter(points, 1e-4)(points);
  n_points = points.rows();

  valuesd values = valuesd::Zero(n_points);
  for (index_t i = 0; i < n_points; i++) {
    auto p = points.row(i);
    Point ap = p * aniso.transpose();
    for (auto j = 0; j < Dim; j++) {
      values(i) += std::sin(ap(j));
    }
  }

  return {std::move(points), std::move(values)};
}

template <int Dim>
std::pair<polatory::geometry::pointNd<Dim>, polatory::geometry::vectorNd<Dim>> sample_grad_data(
    polatory::index_t n_points, const polatory::geometry::matrixNd<Dim>& aniso) {
  using polatory::index_t;
  using polatory::point_cloud::distance_filter;
  using Point = polatory::geometry::pointNd<Dim>;
  using Points = polatory::geometry::pointsNd<Dim>;
  using Vectors = polatory::geometry::vectorsNd<Dim>;

  Points points = Points::Random(n_points, Dim);
  points = distance_filter(points, 1e-4)(points);
  n_points = points.rows();

  Vectors grads(n_points, Dim);
  for (index_t i = 0; i < n_points; i++) {
    auto p = points.row(i);
    Point ap = p * aniso.transpose();
    for (auto j = 0; j < Dim; j++) {
      grads(i, j) = std::cos(ap(j));
    }
    grads.row(i) *= aniso;
  }

  return {std::move(points), std::move(grads)};
}
