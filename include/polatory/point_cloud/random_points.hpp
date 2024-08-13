#pragma once

#include <Eigen/Core>
#include <polatory/geometry/cuboid3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/types.hpp>
#include <random>

namespace polatory::point_cloud {

using seed_type = std::random_device::result_type;

geometry::Points3 random_points(const geometry::Cuboid3& cuboid, Index n,
                                seed_type seed = std::random_device()());

geometry::Points3 random_points(const geometry::Sphere3& sphere, Index n,
                                seed_type seed = std::random_device()());

}  // namespace polatory::point_cloud
