#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/types.hpp>

using polatory::index_t;
using polatory::geometry::sphere3d;
using polatory::geometry::vector3d;
using polatory::point_cloud::normal_estimator;
using polatory::point_cloud::random_points;

TEST(normal_estimator, knn) {
  const auto n_points = index_t{4096};
  const auto k = index_t{10};
  auto points = random_points(sphere3d(), n_points);
  vector3d direction(0.0, 0.0, 1.0);

  auto normals = normal_estimator(points)
                     .estimate_with_knn(k)
                     .orient_toward_direction(direction)
                     .into_normals();

  for (auto n : normals.rowwise()) {
    if (n.norm() == 0.0) continue;

    ASSERT_NEAR(1.0, n.norm(), 1e-14);
    ASSERT_GT(n.dot(direction), 0.0);
  }
}

TEST(normal_estimator, radius) {
  const auto n_points = index_t{4096};
  const auto search_radius = 0.1;
  auto points = random_points(sphere3d(), n_points);
  vector3d direction(0.0, 0.0, 1.0);

  auto normals = normal_estimator(points)
                     .estimate_with_radius(search_radius)
                     .orient_toward_direction(direction)
                     .into_normals();

  for (auto n : normals.rowwise()) {
    if (n.norm() == 0.0) continue;

    ASSERT_NEAR(1.0, n.norm(), 1e-14);
    ASSERT_GT(n.dot(direction), 0.0);
  }
}
