#pragma once

#include <Eigen/Geometry>
#include <cmath>
#include <polatory/common/pi.hpp>
#include <polatory/geometry/point3d.hpp>
#include <random>

using seed_type = std::random_device::result_type;

inline polatory::geometry::linear_transformation3d random_anisotropy(
    seed_type seed = std::random_device()()) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> angle_dist(0.0, 2.0 * polatory::common::pi<double>());
  std::uniform_real_distribution<> log_scale_dist(std::log10(0.1), std::log10(10.0));

  return polatory::geometry::to_linear_transformation3d(
      Eigen::Scaling(std::pow(10.0, log_scale_dist(gen)), std::pow(10.0, log_scale_dist(gen)),
                     std::pow(10.0, log_scale_dist(gen))) *
      Eigen::AngleAxisd(angle_dist(gen), polatory::geometry::vector3d::UnitZ()) *
      Eigen::AngleAxisd(angle_dist(gen), polatory::geometry::vector3d::UnitX()) *
      Eigen::AngleAxisd(angle_dist(gen), polatory::geometry::vector3d::UnitZ()));
}
