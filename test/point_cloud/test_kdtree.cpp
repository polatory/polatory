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
  const size_t n_points = 1024;
  const double radius = 1.0;
  const Eigen::Vector3d center(0.0, 0.0, 0.0);

  const Eigen::Vector3d query_point = center + Eigen::Vector3d(radius, 0.0, 0.0);
  const int k = 10;
  const auto search_radius = 0.1;

  auto points = random_points(sphere3d(center, radius), n_points);

  kdtree tree(points, true);

  std::vector<size_t> indices(42);
  std::vector<double> distances(42);
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

TEST(kdtree, zero_points) {
  const Eigen::Vector3d query_point = Eigen::Vector3d::Zero();
  const int k = 10;
  const auto search_radius = 0.1;

  std::vector<Eigen::Vector3d> points;

  kdtree tree(points, true);

  std::vector<size_t> indices(42);
  std::vector<double> distances(42);
  tree.knn_search(query_point, k, indices, distances);

  ASSERT_EQ(0u, indices.size());
  ASSERT_EQ(0u, distances.size());

  tree.radius_search(query_point, search_radius, indices, distances);

  ASSERT_EQ(0u, indices.size());
  ASSERT_EQ(0u, distances.size());
}
