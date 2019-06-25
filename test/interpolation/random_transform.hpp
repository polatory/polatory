// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <random>

#include <polatory/common/pi.hpp>
#include <polatory/geometry/affine_transformation3d.hpp>

using seed_type = std::random_device::result_type;

inline
polatory::geometry::affine_transformation3d random_transform(seed_type seed = std::random_device()()) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> angle_dist(0.0, 2.0 * polatory::common::pi<double>());
  std::uniform_real_distribution<> scale_dist(0.5, 2.0);
  std::uniform_real_distribution<> log_trans_dist(0.0, 6.0);

  auto r = polatory::geometry::affine_transformation3d::roll_pitch_yaw({ angle_dist(gen), angle_dist(gen), angle_dist(gen) });
  auto s = polatory::geometry::affine_transformation3d::scaling({ scale_dist(gen), scale_dist(gen), scale_dist(gen) });
  auto t = polatory::geometry::affine_transformation3d::translation({ std::pow(log_trans_dist(gen), 10.0), std::pow(log_trans_dist(gen), 10.0), std::pow(log_trans_dist(gen), 10.0) });

  return t * r * s;
}
