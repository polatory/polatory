// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "distribution_generator/spherical_distribution.hpp"
#include "point_cloud/kdtree.hpp"

using namespace polatory::point_cloud;
using polatory::distribution_generator::spherical_distribution;

TEST(kdtree, trivial) {
  size_t n_points = 1024;
  double radius = 1.0;
  Eigen::Vector3d center(0.0, 0.0, 0.0);

  Eigen::Vector3d query_point = center + Eigen::Vector3d(radius, 0.0, 0.0);
  int k = 10;
  auto search_radius = 0.1;
  std::vector<size_t> indices;
  std::vector<double> distances;

  auto points = spherical_distribution(n_points, center, radius);

  kdtree tree(points);

  auto found = tree.knn_search(query_point, k, indices, distances);

  ASSERT_EQ(found, k);
  ASSERT_EQ(found, indices.size());
  ASSERT_EQ(found, distances.size());

  std::sort(indices.begin(), indices.end());
  ASSERT_TRUE(std::unique(indices.begin(), indices.end()) == indices.end());

  found = tree.radius_search(query_point, search_radius, indices, distances);

  ASSERT_EQ(found, indices.size());
  ASSERT_EQ(found, distances.size());
  for (auto distance : distances) {
    ASSERT_LE(distance, search_radius);
  }

  std::sort(indices.begin(), indices.end());
  ASSERT_TRUE(std::unique(indices.begin(), indices.end()) == indices.end());
}
