#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/point_cloud/sdf_data_generator.hpp>
#include <polatory/types.hpp>

using polatory::Index;
using polatory::VecX;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::geometry::Sphere3;
using polatory::geometry::Vectors3;
using polatory::point_cloud::KdTree;
using polatory::point_cloud::random_points;
using polatory::point_cloud::SdfDataGenerator;

TEST(sdf_data_generator, trivial) {
  const auto n_points = Index{512};
  const auto min_distance = 1e-2;
  const auto max_distance = 5e-1;

  Points3 points = random_points(Sphere3(), n_points);
  Vectors3 normals =
      (points + random_points(Sphere3(Point3::Zero(), 0.1), n_points)).rowwise().normalized();

  SdfDataGenerator sdf_data(points, normals, min_distance, max_distance);
  Points3 sdf_points = sdf_data.sdf_points();
  VecX sdf_values = sdf_data.sdf_values();

  EXPECT_EQ(sdf_points.rows(), sdf_values.rows());

  KdTree tree(points);

  std::vector<Index> indices;
  std::vector<double> distances;

  auto n_sdf_points = sdf_points.rows();
  for (Index i = 0; i < n_sdf_points; i++) {
    Point3 sdf_point = sdf_points.row(i);
    auto sdf_value = sdf_values(i);

    tree.knn_search(sdf_point, 1, indices, distances);
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
