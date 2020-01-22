// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <random>

#include <Eigen/Core>

#include <polatory/common/pi.hpp>
#include <polatory/geometry/transformation.hpp>

using seed_type = std::random_device::result_type;

inline
Eigen::Matrix3d random_anisotropy(seed_type seed = std::random_device()()) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> angle_dist(0.0, 2.0 * polatory::common::pi<double>());
  std::uniform_real_distribution<> log_scale_dist(std::log10(0.1), std::log10(10.0));

  return
    polatory::geometry::scaling({ std::pow(10.0, log_scale_dist(gen)), std::pow(10.0, log_scale_dist(gen)), std::pow(10.0, log_scale_dist(gen)) }) *
    polatory::geometry::rotation(angle_dist(gen), 2) *
    polatory::geometry::rotation(angle_dist(gen), 0) *
    polatory::geometry::rotation(angle_dist(gen), 2);
}
