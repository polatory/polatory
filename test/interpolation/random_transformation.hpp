// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <random>

#include <polatory/common/pi.hpp>
#include <polatory/geometry/affine_transformation3d.hpp>

using seed_type = std::random_device::result_type;

inline
polatory::geometry::affine_transformation3d random_transformation(seed_type seed = std::random_device()()) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> angle_dist(0.0, 2.0 * polatory::common::pi<double>());
  std::uniform_real_distribution<> log_scale_dist(std::log10(0.1), std::log10(10.0));

  auto r = polatory::geometry::affine_transformation3d::roll_pitch_yaw({ angle_dist(gen), angle_dist(gen), angle_dist(gen) });
  auto s = polatory::geometry::affine_transformation3d::scaling({ std::pow(10.0, log_scale_dist(gen)), std::pow(10.0, log_scale_dist(gen)), std::pow(10.0, log_scale_dist(gen)) });

  return r * s;
}
