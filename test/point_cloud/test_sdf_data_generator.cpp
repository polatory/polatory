#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>
#include <polatory/types.hpp>

using polatory::index_t;
using polatory::common::valuesd;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::sphere3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::kdtree;
using polatory::point_cloud::random_points;
using polatory::point_cloud::sdf_data_generator;

TEST(sdf_data_generator, trivial) {
  const auto n_points = index_t{512};
  const auto min_distance = 1e-2;
  const auto max_distance = 5e-1;

  points3d points = random_points(sphere3d(), n_points);
  vectors3d normals =
      (points + random_points(sphere3d(point3d::Zero(), 0.1), n_points)).rowwise().normalized();

  sdf_data_generator sdf_data(points, normals, min_distance, max_distance);
  points3d sdf_points = sdf_data.sdf_points();
  valuesd sdf_values = sdf_data.sdf_values();

  EXPECT_EQ(sdf_points.rows(), sdf_values.rows());

  kdtree tree(points, true);

  auto n_sdf_points = sdf_points.rows();
  for (index_t i = 0; i < n_sdf_points; i++) {
    point3d sdf_point = sdf_points.row(i);
    auto sdf_value = sdf_values(i);

    auto [indices, distances] = tree.knn_search(sdf_point, 1);
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
