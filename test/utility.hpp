#pragma once

#include <Eigen/Core>
#include <cmath>
#include <numbers>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <utility>

template <int Dim>
polatory::geometry::matrixNd<Dim> random_anisotropy() {
  using polatory::common::orthonormalize_cols;
  using Vector = polatory::geometry::vectorNd<Dim>;
  using Matrix = polatory::geometry::matrixNd<Dim>;

  // Rotation.
  Matrix a = Matrix::Random();
  orthonormalize_cols(a);
  if (a.determinant() < 0.0) {
    a.col(0) *= -1.0;
  }

  // Scaling.
  a.diagonal().array() *= pow(10.0, Vector::Random().array());

  return a;
}

template <int Dim>
std::pair<polatory::geometry::pointsNd<Dim>, polatory::common::valuesd> sample_data(
    polatory::index_t& n_points, const polatory::geometry::matrixNd<Dim>& aniso) {
  using polatory::index_t;
  using polatory::common::valuesd;
  using polatory::point_cloud::distance_filter;
  using Point = polatory::geometry::pointNd<Dim>;
  using Points = polatory::geometry::pointsNd<Dim>;

  Points points = Points::Random(n_points, Dim);
  points = distance_filter(points, 1e-6)(points);
  n_points = points.rows();

  valuesd values = valuesd::Zero(n_points);
  for (index_t i = 0; i < n_points; i++) {
    auto p = points.row(i);
    Point ap = p * aniso.transpose();
    for (auto j = 0; j < Dim; j++) {
      values(i) += std::sin(std::numbers::pi * ap(j));
    }
  }

  return {std::move(points), std::move(values)};
}

template <int Dim>
std::pair<polatory::geometry::pointsNd<Dim>, polatory::geometry::vectorsNd<Dim>> sample_grad_data(
    polatory::index_t& n_points, const polatory::geometry::matrixNd<Dim>& aniso) {
  using polatory::index_t;
  using polatory::point_cloud::distance_filter;
  using Point = polatory::geometry::pointNd<Dim>;
  using Points = polatory::geometry::pointsNd<Dim>;
  using Vectors = polatory::geometry::vectorsNd<Dim>;

  Points points = Points::Random(n_points, Dim);
  points = distance_filter(points, 1e-6)(points);
  n_points = points.rows();

  Vectors grads(n_points, Dim);
  for (index_t i = 0; i < n_points; i++) {
    auto p = points.row(i);
    Point ap = p * aniso.transpose();
    for (auto j = 0; j < Dim; j++) {
      grads(i, j) = std::numbers::pi * std::cos(std::numbers::pi * ap(j));
    }
    grads.row(i) *= aniso;
  }

  return {std::move(points), std::move(grads)};
}
