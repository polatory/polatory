// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/normal_estimator.hpp>
#include <polatory/point_cloud/random_points.hpp>

using polatory::common::row_range;
using polatory::geometry::sphere3d;
using polatory::geometry::vector3d;
using polatory::point_cloud::normal_estimator;
using polatory::point_cloud::random_points;

TEST(normal_estimator, knn) {
  const size_t n_points = 4096;
  const size_t k = 10;
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
  const size_t n_points = 4096;
  const double search_radius = 0.1;
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
