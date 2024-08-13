#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/types.hpp>

using polatory::Index;
using polatory::geometry::Sphere3;
using polatory::geometry::Vector3;
using polatory::point_cloud::NormalEstimator;
using polatory::point_cloud::random_points;

TEST(normal_estimator, knn) {
  const auto n_points = Index{4096};
  const auto k = Index{10};
  auto points = random_points(Sphere3(), n_points);
  Vector3 direction(0.0, 0.0, 1.0);

  auto normals = NormalEstimator(points)
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
  const auto n_points = Index{4096};
  const auto search_radius = 0.1;
  auto points = random_points(Sphere3(), n_points);
  Vector3 direction(0.0, 0.0, 1.0);

  auto normals = NormalEstimator(points)
                     .estimate_with_radius(search_radius)
                     .orient_toward_direction(direction)
                     .into_normals();

  for (auto n : normals.rowwise()) {
    if (n.norm() == 0.0) continue;

    ASSERT_NEAR(1.0, n.norm(), 1e-14);
    ASSERT_GT(n.dot(direction), 0.0);
  }
}
