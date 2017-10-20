// Copyright (c) 2016, GSI and The Polatory Authors.

#include <string>

#include <Eigen/Core>

#include "polatory/io/write_table.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/point_cloud/random_points.hpp"

using polatory::geometry::sphere3d;
using polatory::io::write_points;
using polatory::numeric::to_string;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::random_points;

int main(int argc, char *argv[]) {
  auto n_points = std::stoi(argv[1]);
  auto seed = std::stoi(argv[2]);

  auto points = distance_filter(
    random_points(sphere3d(), n_points, seed),
    1e-8).filtered_points();

  write_points(argv[1], points);

  return 0;
}
