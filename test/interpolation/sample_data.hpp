#pragma once

#include <algorithm>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>
#include <polatory/types.hpp>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

inline std::pair<polatory::geometry::points3d, polatory::common::valuesd> sample_numerical_data(
    polatory::index_t n_points) {
  using polatory::common::valuesd;
  using polatory::geometry::points3d;
  using polatory::geometry::sphere3d;
  using polatory::point_cloud::distance_filter;
  using polatory::point_cloud::random_points;

  points3d points = random_points(sphere3d(), n_points);
  valuesd values = valuesd::Random(n_points);

  std::tie(points, values) = distance_filter(points, 1e-4)(points, values);

  return {std::move(points), std::move(values)};
}

inline std::pair<polatory::geometry::points3d, polatory::common::valuesd> sample_sdf_data(
    polatory::index_t n_surface_points) {
  using polatory::index_t;
  using polatory::common::valuesd;
  using polatory::geometry::points3d;
  using polatory::geometry::sphere3d;
  using polatory::point_cloud::distance_filter;
  using polatory::point_cloud::random_points;
  using polatory::point_cloud::sdf_data_generator;

  points3d surface_points = random_points(sphere3d(), n_surface_points);

  sdf_data_generator sdf_data(surface_points, surface_points, 1e-3, 1e-2);
  points3d points = sdf_data.sdf_points();
  valuesd values = sdf_data.sdf_values();

  std::vector<index_t> indices(points.rows());
  std::iota(indices.begin(), indices.end(), index_t{0});
  std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

  points = points(indices, Eigen::all).eval();
  values = values(indices, Eigen::all).eval();

  std::tie(points, values) = distance_filter(points, 1e-4)(points, values);

  return {std::move(points), std::move(values)};
}
