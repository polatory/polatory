#include <gtest/gtest.h>

#include <jizai/geometry/point3d.hpp>
#include <jizai/point_cloud/kdtree.hpp>
#include <jizai/point_cloud/random_points.hpp>
#include <jizai/point_cloud/sdf_data_generator.hpp>
#include <jizai/types.hpp>

using jizai::Index;
using jizai::VecX;
using jizai::geometry::Point3;
using jizai::geometry::Points3;
using jizai::geometry::Sphere3;
using jizai::geometry::Vectors3;
using jizai::point_cloud::KdTree;
using jizai::point_cloud::random_points;
using jizai::point_cloud::SdfDataGenerator;

TEST(sdf_data_generator, trivial) {
  const auto n_points = Index{512};
  const auto offset = 1e-2;

  Points3 points = random_points(Sphere3(), n_points);
  Vectors3 normals =
      (points + random_points(Sphere3(Point3::Zero(), 0.1), n_points)).rowwise().normalized();

  SdfDataGenerator sdf_data(points, normals, offset);
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

      EXPECT_LE(std::abs(sdf_value), offset);
      EXPECT_GT(sdf_value * normal.dot(sdf_point - point), 0.0);
    }
  }
}
