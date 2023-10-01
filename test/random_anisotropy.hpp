#pragma once

#include <Eigen/Geometry>
#include <cmath>
#include <numbers>
#include <polatory/geometry/point3d.hpp>
#include <random>

using seed_type = std::random_device::result_type;

inline polatory::geometry::matrix3d random_anisotropy(seed_type seed = std::random_device()()) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> angle_dist(0.0, 2.0 * std::numbers::pi);
  std::uniform_real_distribution<> log_scale_dist(std::log10(0.1), std::log10(10.0));

  return polatory::geometry::to_matrix3d(
      Eigen::Scaling(std::pow(10.0, log_scale_dist(gen)), std::pow(10.0, log_scale_dist(gen)),
                     std::pow(10.0, log_scale_dist(gen))) *
      Eigen::AngleAxisd(angle_dist(gen), polatory::geometry::vector3d::UnitZ()) *
      Eigen::AngleAxisd(angle_dist(gen), polatory::geometry::vector3d::UnitX()) *
      Eigen::AngleAxisd(angle_dist(gen), polatory::geometry::vector3d::UnitZ()));
}
