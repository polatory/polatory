// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/point_cloud/random_points.hpp>

#include <cmath>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/exception.hpp>

namespace polatory {
namespace point_cloud {

geometry::points3d random_points(const geometry::cuboid3d& cuboid,
    index_t n,
    seed_type seed) {
  auto size = cuboid.max() - cuboid.min();
  if (size.minCoeff() <= 0.0)
    throw common::invalid_argument("cuboid must be a valid region");

  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  geometry::points3d points(n, 3);

  for (auto p : common::row_range(points)) {
    auto x = dist(gen) * size(0);
    auto y = dist(gen) * size(1);
    auto z = dist(gen) * size(2);

    p = cuboid.min() + geometry::vector3d(x, y, z);
  }

  return points;
}

// See Marsaglia (1972) at:
// http://mathworld.wolfram.com/SpherePointPicking.html
geometry::points3d random_points(const geometry::sphere3d& sphere,
    index_t n,
    seed_type seed) {
  if (sphere.radius() <= 0.0)
    throw common::invalid_argument("sphere must be a valid region");

  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  geometry::points3d points(n, 3);

  for (auto p : common::row_range(points)) {
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

    p = sphere.center() + sphere.radius() * geometry::vector3d(x, y, z);
  }

  return points;
}

}  // namespace point_cloud
}  // namespace polatory
