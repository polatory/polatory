#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/types.hpp>

using polatory::Index;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::geometry::Sphere3;
using polatory::geometry::Vector3;
using polatory::point_cloud::KdTree;
using polatory::point_cloud::random_points;

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

    std::sort(indices.begin(), indices.end());
    EXPECT_EQ(indices.end(), std::unique(indices.begin(), indices.end()));
  }

  {
    tree.radius_search(query_point, search_radius, indices, distances);

    EXPECT_EQ(indices.size(), distances.size());
    for (auto distance : distances) {
      EXPECT_LE(distance, search_radius);
    }

    std::sort(indices.begin(), indices.end());
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
