// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <random>
#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace random_points {

inline std::vector<Eigen::Vector3d> box_points(size_t n, const Eigen::Vector3d& center, double radius) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-radius, radius);

  std::vector<Eigen::Vector3d> points;

  for (size_t i = 0; i < n; i++) {
    auto x = dist(gen);
    auto y = dist(gen);
    auto z = dist(gen);

    points.push_back(Eigen::Vector3d(x, y, z) + center);
  }

  return points;
}

} // namespace random_points
} // namespace polatory
