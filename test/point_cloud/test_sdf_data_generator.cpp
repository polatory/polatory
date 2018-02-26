// Copyright (c) 2016, GSI and The Polatory Authors.

#include <tuple>

#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>

using polatory::common::valuesd;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::sphere3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::kdtree;
using polatory::point_cloud::random_points;
using polatory::point_cloud::sdf_data_generator;

TEST(sdf_data_generator, trivial) {
  size_t n_points = 512;
  auto min_distance = 1e-2;
  auto max_distance = 5e-1;

  points3d points = random_points(sphere3d(), n_points);
  vectors3d normals = (points + random_points(sphere3d(point3d::Zero(), 0.1), n_points))
    .rowwise().normalized();

  sdf_data_generator sdf_data(points, normals, min_distance, max_distance);
  points3d sdf_points = sdf_data.sdf_points();
  valuesd sdf_values = sdf_data.sdf_values();

  EXPECT_EQ(sdf_points.rows(), sdf_values.rows());

  kdtree tree(points, true);

  std::vector<size_t> indices;
  std::vector<double> distances;

  for (size_t i = 0; i < sdf_points.rows(); i++) {
    point3d sdf_point = sdf_points.row(i);
    auto sdf_value = sdf_values(i);

    std::tie(indices, distances) = tree.knn_search(sdf_point, 1);
    EXPECT_NEAR(distances[0], std::abs(sdf_value), 1e-15);

    if (sdf_values(i) != 0.0) {
      auto point = points.row(indices[0]);
      auto normal = normals.row(indices[0]);

      EXPECT_GE(std::abs(sdf_value), min_distance);
      EXPECT_LE(std::abs(sdf_value), max_distance);
      EXPECT_GT(sdf_value * normal.dot(sdf_point - point), 0.0);
    }
  }
}
