// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <random>

#include <Eigen/Core>

#include <polatory/geometry/cuboid3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>

namespace polatory {
namespace point_cloud {

using seed_type = std::random_device::result_type;

geometry::points3d random_points(const geometry::cuboid3d& cuboid,
                                 size_t n,
                                 seed_type seed = std::random_device()());

geometry::points3d random_points(const geometry::sphere3d& sphere,
                                 size_t n,
                                 seed_type seed = std::random_device()());

} // namespace point_cloud
} // namespace polatory
