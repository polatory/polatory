#pragma once

#include <Eigen/Core>
#include <cmath>
#include <numbers>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <utility>

template <int Dim>
polatory::geometry::matrixNd<Dim> random_rotation() {
  using polatory::common::orthonormalize_cols;
  using Matrix = polatory::geometry::matrixNd<Dim>;

  Matrix rot = Matrix::Random();
  orthonormalize_cols(rot);
  if (rot.determinant() < 0.0) {
    rot.col(0) *= -1.0;
  }

  return rot;
}

template <int Dim>
polatory::geometry::matrixNd<Dim> random_scaling() {
  using Vector = polatory::geometry::vectorNd<Dim>;
  using Matrix = polatory::geometry::matrixNd<Dim>;

  Matrix scale = Matrix::Identity();
  scale.diagonal().array() *= pow(10.0, Vector::Random().array());

  return scale;
}

template <int Dim>
polatory::geometry::matrixNd<Dim> random_anisotropy() {
  return random_scaling<Dim>() * random_rotation<Dim>();
}

template <int Dim>
std::pair<polatory::geometry::pointsNd<Dim>, polatory::vectord> sample_data(
    polatory::index_t& n_points, const polatory::geometry::matrixNd<Dim>& aniso) {
  using polatory::index_t;
  using polatory::vectord;
  using polatory::point_cloud::distance_filter;
  using Point = polatory::geometry::pointNd<Dim>;
  using Points = polatory::geometry::pointsNd<Dim>;

  Points points = Points::Random(n_points, Dim);
  points = distance_filter(points, 1e-6)(points);
  n_points = points.rows();

  vectord values = vectord::Zero(n_points);
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
