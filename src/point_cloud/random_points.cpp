#include <cmath>
#include <polatory/point_cloud/random_points.hpp>
#include <stdexcept>

namespace polatory::point_cloud {

geometry::Points3 random_points(const geometry::Cuboid3& cuboid, Index n, seed_type seed) {
  auto size = cuboid.max() - cuboid.min();
  if (!(size.minCoeff() > 0.0)) {
    throw std::invalid_argument("cuboid must have a positive volume");
  }

  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  geometry::Points3 points(n, 3);

  for (auto p : points.rowwise()) {
    auto x = dist(gen) * size(0);
    auto y = dist(gen) * size(1);
    auto z = dist(gen) * size(2);

    p = cuboid.min() + geometry::Vector3(x, y, z);
  }

  return points;
}

// See Marsaglia (1972) at:
// http://mathworld.wolfram.com/SpherePointPicking.html
geometry::Points3 random_points(const geometry::Sphere3& sphere, Index n, seed_type seed) {
  if (!(sphere.radius() > 0.0)) {
    throw std::invalid_argument("sphere must have a positive volume");
  }

  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  geometry::Points3 points(n, 3);

  for (auto p : points.rowwise()) {
    double x1{};
    double x2{};
    double rsq{};
    do {
      x1 = dist(gen);
      x2 = dist(gen);
      rsq = x1 * x1 + x2 * x2;
    } while (rsq >= 1.0);

    auto x = 2.0 * x1 * std::sqrt(1.0 - rsq);
    auto y = 2.0 * x2 * std::sqrt(1.0 - rsq);
    auto z = 1.0 - 2.0 * rsq;

    p = sphere.center() + sphere.radius() * geometry::Vector3(x, y, z);
  }

  return points;
}

}  // namespace polatory::point_cloud
