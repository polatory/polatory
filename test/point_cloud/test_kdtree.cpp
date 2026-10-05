#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <jizai/geometry/point3d.hpp>
#include <jizai/point_cloud/kdtree.hpp>
#include <jizai/point_cloud/random_points.hpp>
#include <jizai/types.hpp>

using jizai::Index;
using jizai::geometry::Point3;
using jizai::geometry::Points3;
using jizai::geometry::Sphere3;
using jizai::geometry::Vector3;
using jizai::point_cloud::KdTree;
using jizai::point_cloud::random_points;

TEST(kdtree, trivial) {
  const auto n_points = Index{1024};
  const auto radius = 1.0;
  const Point3 center(0.0, 0.0, 0.0);

  const Point3 query_point = center + Vector3(radius, 0.0, 0.0);
  const auto k = Index{10};
  const auto search_radius = 0.1;

  auto points = random_points(Sphere3(center, radius), n_points);

  KdTree tree(points);

  std::vector<Index> indices;
  std::vector<double> distances;

  {
    tree.knn_search(query_point, k, indices, distances);

    EXPECT_EQ(k, indices.size());
    EXPECT_EQ(indices.size(), distances.size());

    std::ranges::sort(indices);
    EXPECT_EQ(indices.end(), std::unique(indices.begin(), indices.end()));
  }

  {
    tree.radius_search(query_point, search_radius, indices, distances);

    EXPECT_EQ(indices.size(), distances.size());
    for (auto distance : distances) {
      EXPECT_LE(distance, search_radius);
    }

    std::ranges::sort(indices);
    EXPECT_EQ(indices.end(), std::unique(indices.begin(), indices.end()));
  }
}

TEST(kdtree, zero_points) {
  const Point3 query_point = Point3::Zero();
  const auto k = Index{10};
  const auto search_radius = 0.1;

  Points3 points;

  KdTree tree(points);

  std::vector<Index> indices;
  std::vector<double> distances;

  {
    tree.knn_search(query_point, k, indices, distances);

    EXPECT_EQ(0u, indices.size());
    EXPECT_EQ(0u, distances.size());
  }

  {
    tree.radius_search(query_point, search_radius, indices, distances);

    EXPECT_EQ(0u, indices.size());
    EXPECT_EQ(0u, distances.size());
  }
}
