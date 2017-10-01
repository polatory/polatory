// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace polatory {
namespace random_points {

// See Marsaglia (1972) at:
// http://mathworld.wolfram.com/SpherePointPicking.html
inline std::vector<Eigen::Vector3d> sphere_points(size_t n, const Eigen::Vector3d& center = Eigen::Vector3d::Zero(), double radius = 1.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  std::vector<Eigen::Vector3d> points;

  for (size_t i = 0; i < n; i++) {
    double x1;
    double x2;
    double rsq;
    do {
      x1 = dist(gen);
      x2 = dist(gen);
      rsq = x1 * x1 + x2 * x2;
    } while (rsq >= 1.0);

    auto x = 2.0 * x1 * std::sqrt(1.0 - rsq);
    auto y = 2.0 * x2 * std::sqrt(1.0 - rsq);
    auto z = 1.0 - 2.0 * rsq;

    points.push_back(radius * Eigen::Vector3d(x, y, z) + center);
  }

  return points;
}

} // namespace random_points
} // namespace polatory
