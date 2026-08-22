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
polatory::Mat<Dim> random_rotation() {
  using polatory::common::orthonormalize_cols;
  using Mat = polatory::Mat<Dim>;

  Mat rot = Mat::Random();
  orthonormalize_cols(rot);
  if (rot.determinant() < 0.0) {
    rot.col(0) *= -1.0;
  }

  return rot;
}

template <int Dim>
polatory::Mat<Dim> random_scaling() {
  using Mat = polatory::Mat<Dim>;
  using Vector = polatory::geometry::Vector<Dim>;

  Mat scale = Mat::Identity();
  scale.diagonal().array() *= pow(10.0, 0.5 * Vector::Random().array());
  // Normalize the determinant.
  scale.diagonal() *= std::pow(scale.diagonal().prod(), -1.0 / Dim);

  return scale;
}

template <int Dim>
polatory::Mat<Dim> random_anisotropy() {
  return random_scaling<Dim>() * random_rotation<Dim>();
}

template <int Dim>
std::pair<polatory::geometry::Points<Dim>, polatory::VecX> sample_data(
    polatory::Index& n_points, const polatory::Mat<Dim>& aniso) {
  using polatory::Index;
  using polatory::VecX;
  using polatory::geometry::transform_points;
  using polatory::point_cloud::DistanceFilter;
  using Mat = polatory::Mat<Dim>;
  using Points = polatory::geometry::Points<Dim>;

  Points a_points = Points::Random(n_points, Dim);
  a_points = DistanceFilter(a_points).filter(1e-6)(a_points);
  n_points = a_points.rows();

  Mat aniso_inv = aniso.inverse();
  Points points = transform_points<Dim>(aniso_inv, a_points);

  VecX values = VecX::Zero(n_points);
  for (Index i = 0; i < n_points; i++) {
    auto ap = a_points.row(i);
    for (auto j = 0; j < Dim; j++) {
      values(i) += std::sin(std::numbers::pi * ap(j));
    }
  }

  return {std::move(points), std::move(values)};
}

template <int Dim>
std::pair<polatory::geometry::Points<Dim>, polatory::geometry::Vectors<Dim>> sample_grad_data(
    polatory::Index& n_points, const polatory::Mat<Dim>& aniso) {
  using polatory::Index;
  using polatory::geometry::transform_points;
  using polatory::point_cloud::DistanceFilter;
  using Mat = polatory::Mat<Dim>;
  using Points = polatory::geometry::Points<Dim>;
  using Vectors = polatory::geometry::Vectors<Dim>;

  Points a_points = Points::Random(n_points, Dim);
  a_points = DistanceFilter(a_points).filter(1e-6)(a_points);
  n_points = a_points.rows();

  Mat aniso_inv = aniso.inverse();
  Points points = transform_points<Dim>(aniso_inv, a_points);

  Vectors grads(n_points, Dim);
  for (Index i = 0; i < n_points; i++) {
    auto ap = a_points.row(i);
    for (auto j = 0; j < Dim; j++) {
      grads(i, j) = std::numbers::pi * std::cos(std::numbers::pi * ap(j));
    }
    grads.row(i) *= aniso;
  }

  return {std::move(points), std::move(grads)};
}
