// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/point_cloud/kdtree.hpp"
#include "polatory/point_cloud/random_points.hpp"

using namespace polatory::point_cloud;
using polatory::geometry::sphere3d;
using polatory::point_cloud::random_points;

TEST(kdtree, trivial) {
  size_t n_points = 1024;
  double radius = 1.0;
  Eigen::Vector3d center(0.0, 0.0, 0.0);

  Eigen::Vector3d query_point = center + Eigen::Vector3d(radius, 0.0, 0.0);
  int k = 10;
  auto search_radius = 0.1;
  std::vector<size_t> indices;
  std::vector<double> distances;

  auto points = random_points(sphere3d(center, radius), n_points);

  kdtree tree(points, true);

  tree.knn_search(query_point, k, indices, distances);

  ASSERT_EQ(k, indices.size());
  ASSERT_EQ(indices.size(), distances.size());

  std::sort(indices.begin(), indices.end());
  ASSERT_EQ(indices.end(), std::unique(indices.begin(), indices.end()));

  tree.radius_search(query_point, search_radius, indices, distances);

  ASSERT_EQ(indices.size(), distances.size());
  for (auto distance : distances) {
    ASSERT_LE(distance, search_radius);
  }

  std::sort(indices.begin(), indices.end());
  ASSERT_EQ(indices.end(), std::unique(indices.begin(), indices.end()));
}
