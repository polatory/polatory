#pragma once

#include <Eigen/Core>
#include <polatory/geometry/cuboid3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/types.hpp>
#include <random>

namespace polatory {
namespace point_cloud {

using seed_type = std::random_device::result_type;

geometry::points3d random_points(const geometry::cuboid3d& cuboid, index_t n,
                                 seed_type seed = std::random_device()());

geometry::points3d random_points(const geometry::sphere3d& sphere, index_t n,
                                 seed_type seed = std::random_device()());

}  // namespace point_cloud
}  // namespace polatory
