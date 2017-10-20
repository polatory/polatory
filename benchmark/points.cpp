// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "polatory/numeric/to_string.hpp"
#include "polatory/point_cloud/distance_filter.hpp"
#include "polatory/point_cloud/random_points.hpp"

using polatory::geometry::sphere3d;
using polatory::numeric::to_string;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::random_points;

int main(int argc, char *argv[]) {
  auto n_points = std::stoi(argv[1]);
  auto seed = std::stoi(argv[2]);

  auto points = distance_filter(
    random_points(sphere3d(), n_points, seed),
    1e-8).filtered_points();

  for (const auto& p : points) {
    std::cout << to_string(p(0)) << ' ' << to_string(p(1)) << ' ' << to_string(p(2)) << std::endl;
  }

  return 0;
}
