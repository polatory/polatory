#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/point_cloud/offset_points_generator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/types.hpp>

using polatory::index_t;
using polatory::common::valuesd;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::sphere3d;
using polatory::geometry::vector3d;
using polatory::geometry::vectors3d;
using polatory::point_cloud::kdtree;
using polatory::point_cloud::offset_points_generator;
using polatory::point_cloud::random_points;

TEST(offset_points_generator, trivial) {
  const auto n_points = index_t{512};

  points3d points = random_points(sphere3d(), n_points);
  vectors3d normals =
      (points + random_points(sphere3d(point3d::Zero(), 0.1), n_points)).rowwise().normalized();
  valuesd offsets = valuesd::Random(n_points);

  offset_points_generator off(points, normals, offsets);
  points3d new_points = off.new_points();
  vectors3d new_normals = off.new_normals();

  EXPECT_EQ(new_points.rows(), new_normals.rows());

  kdtree tree(points, true);

  std::vector<index_t> nn_indices;
  std::vector<double> nn_distances;

  auto n_new_points = new_points.rows();
  for (index_t i = 0; i < n_new_points; i++) {
    point3d new_point = new_points.row(i);
    vector3d new_normal = new_normals.row(i);

    tree.knn_search(new_point, 1, nn_indices, nn_distances);

    auto index = nn_indices.at(0);
    point3d point = points.row(index);
    vector3d normal = normals.row(index);
    auto offset = offsets(index);

    EXPECT_NEAR((new_point - point).norm(), std::abs(offset), 1e-15);
    EXPECT_EQ(normal, new_normal);

    if (offset != 0.0) {
      EXPECT_GT(offset * (new_point - point).dot(normal), 0.0);
    }
  }
}
