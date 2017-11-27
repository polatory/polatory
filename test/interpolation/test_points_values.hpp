// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>

namespace {

std::pair<polatory::geometry::points3d, Eigen::VectorXd> test_points_values(size_t n_surface_points) {
  using namespace polatory;

  auto surface_points = point_cloud::random_points(geometry::sphere3d(), n_surface_points);

  point_cloud::sdf_data_generator sdf_data(surface_points, surface_points, 2e-4, 1e-3);
  auto points = sdf_data.sdf_points();
  auto values = sdf_data.sdf_values();

  std::tie(points, values) = point_cloud::distance_filter(points, 1e-6)
    .filtered(points, values);

  return std::make_pair(std::move(points), std::move(values));
}

} // namespace
