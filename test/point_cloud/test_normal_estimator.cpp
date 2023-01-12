#include <gtest/gtest.h>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/types.hpp>

using polatory::common::row_range;
using polatory::geometry::sphere3d;
using polatory::geometry::vector3d;
using polatory::point_cloud::normal_estimator;
using polatory::point_cloud::random_points;
using polatory::index_t;

TEST(normal_estimator, knn) {
  const auto n_points = index_t{ 4096 };
  const auto k = index_t{ 10 };
  auto points = random_points(sphere3d(), n_points);
  vector3d outward_vector(0.0, 0.0, 1.0);

  auto normals = normal_estimator(points)
    .estimate_with_knn(k)
    .orient_by_outward_vector(outward_vector);

  for (auto n : row_range(normals)) {
    if (n.norm() == 0.0)
      continue;

    ASSERT_NEAR(1.0, n.norm(), 1e-14);
    ASSERT_GT(n.dot(outward_vector), 0.0);
  }
}

TEST(normal_estimator, radius) {
  const auto n_points = index_t{ 4096 };
  const auto search_radius = 0.1;
  auto points = random_points(sphere3d(), n_points);
  vector3d outward_vector(0.0, 0.0, 1.0);

  auto normals = normal_estimator(points)
    .estimate_with_radius(search_radius)
    .orient_by_outward_vector(outward_vector);

  for (auto n : row_range(normals)) {
    if (n.norm() == 0.0)
      continue;

    ASSERT_NEAR(1.0, n.norm(), 1e-14);
    ASSERT_GT(n.dot(outward_vector), 0.0);
  }
}
