#include <gtest/gtest.h>

#include <jizai/geometry/point3d.hpp>
#include <jizai/point_cloud/normal_estimator.hpp>
#include <jizai/point_cloud/random_points.hpp>
#include <jizai/types.hpp>

using jizai::Index;
using jizai::geometry::Sphere3;
using jizai::geometry::Vector3;
using jizai::point_cloud::NormalEstimator;
using jizai::point_cloud::random_points;

TEST(normal_estimator, knn) {
  const auto n_points = Index{4096};
  const auto k = Index{10};
  auto points = random_points(Sphere3(), n_points);
  Vector3 direction(0.0, 0.0, 1.0);

  NormalEstimator estimator(points);
  estimator.estimate_with_knn(k);
  estimator.orient_toward_direction(direction);
  const auto& normals = estimator.normals();

  for (auto n : normals.rowwise()) {
    if (n.norm() == 0.0) {
      continue;
    }

    ASSERT_NEAR(1.0, n.norm(), 1e-14);
    ASSERT_GT(n.dot(direction), 0.0);
  }
}

TEST(normal_estimator, radius) {
  const auto n_points = Index{4096};
  const auto search_radius = 0.1;
  auto points = random_points(Sphere3(), n_points);
  Vector3 direction(0.0, 0.0, 1.0);

  NormalEstimator estimator(points);
  estimator.estimate_with_radius(search_radius);
  estimator.orient_toward_direction(direction);
  const auto& normals = estimator.normals();

  for (auto n : normals.rowwise()) {
    if (n.norm() == 0.0) {
      continue;
    }

    ASSERT_NEAR(1.0, n.norm(), 1e-14);
    ASSERT_GT(n.dot(direction), 0.0);
  }
}
