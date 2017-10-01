// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/point_cloud/scattered_data_generator.hpp"
#include "polatory/random_points/sphere_points.hpp"

namespace {

std::pair<std::vector<Eigen::Vector3d>, Eigen::VectorXd> test_points_values(size_t n_surface_points) {
  using namespace polatory;

  auto surface_points = random_points::sphere_points(n_surface_points);
  point_cloud::distance_filter filter(surface_points, 1e-6);
  surface_points = filter.filtered_points();

  point_cloud::scattered_data_generator scatter_gen(surface_points, surface_points, 2e-4, 1e-3);
  auto points = scatter_gen.scattered_points();
  auto values = scatter_gen.scattered_values();

  return std::make_pair(std::move(points), std::move(values));
}

} // namespace
