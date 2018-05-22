// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <tuple>
#include <utility>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>

inline
std::pair<polatory::geometry::points3d, polatory::common::valuesd> sample_numerical_data(size_t n_points) {
  using polatory::common::valuesd;
  using polatory::geometry::points3d;
  using polatory::geometry::sphere3d;
  using polatory::point_cloud::distance_filter;
  using polatory::point_cloud::random_points;

  points3d points = random_points(sphere3d(), n_points);
  valuesd values = valuesd::Random(n_points);

  std::tie(points, values) = distance_filter(points, 1e-6)
    .filtered(points, values);

  return std::make_pair(std::move(points), std::move(values));
}

inline
std::pair<polatory::geometry::points3d, polatory::common::valuesd> sample_sdf_data(size_t n_surface_points) {
  using polatory::common::valuesd;
  using polatory::geometry::points3d;
  using polatory::geometry::sphere3d;
  using polatory::point_cloud::distance_filter;
  using polatory::point_cloud::random_points;
  using polatory::point_cloud::sdf_data_generator;

  points3d surface_points = random_points(sphere3d(), n_surface_points);

  sdf_data_generator sdf_data(surface_points, surface_points, 2e-4, 1e-3);
  points3d points = sdf_data.sdf_points();
  valuesd values = sdf_data.sdf_values();

  std::tie(points, values) = distance_filter(points, 1e-6)
    .filtered(points, values);

  return std::make_pair(std::move(points), std::move(values));
}
