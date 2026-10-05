#pragma once

#include <Eigen/Core>
#include <cmath>
#include <jizai/common/orthonormalize.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/point_cloud/distance_filter.hpp>
#include <jizai/types.hpp>
#include <numbers>
#include <utility>

template <int Dim>
jizai::Mat<Dim> random_rotation() {
  using jizai::common::orthonormalize_cols;
  using Mat = jizai::Mat<Dim>;

  Mat rot = Mat::Random();
  orthonormalize_cols(rot);
  if (rot.determinant() < 0.0) {
    rot.col(0) *= -1.0;
  }

  return rot;
}

template <int Dim>
jizai::Mat<Dim> random_scaling() {
  using Mat = jizai::Mat<Dim>;
  using Vector = jizai::geometry::Vector<Dim>;

  Mat scale = Mat::Identity();
  scale.diagonal().array() *= pow(10.0, 0.5 * Vector::Random().array());
  // Normalize the determinant.
  scale.diagonal() *= std::pow(scale.diagonal().prod(), -1.0 / Dim);

  return scale;
}

template <int Dim>
jizai::Mat<Dim> random_anisotropy() {
  return random_scaling<Dim>() * random_rotation<Dim>();
}

template <int Dim>
std::pair<jizai::geometry::Points<Dim>, jizai::VecX> sample_data(jizai::Index& n_points,
                                                                 const jizai::Mat<Dim>& aniso) {
  using jizai::Index;
  using jizai::VecX;
  using jizai::geometry::transform_points;
  using jizai::point_cloud::DistanceFilter;
  using Mat = jizai::Mat<Dim>;
  using Points = jizai::geometry::Points<Dim>;

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
std::pair<jizai::geometry::Points<Dim>, jizai::geometry::Vectors<Dim>> sample_grad_data(
    jizai::Index& n_points, const jizai::Mat<Dim>& aniso) {
  using jizai::Index;
  using jizai::geometry::transform_points;
  using jizai::point_cloud::DistanceFilter;
  using Mat = jizai::Mat<Dim>;
  using Points = jizai::geometry::Points<Dim>;
  using Vectors = jizai::geometry::Vectors<Dim>;

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
