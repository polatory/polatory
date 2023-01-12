#include <cmath>

#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>

using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::vector3d;
using polatory::point_cloud::plane_estimator;

TEST(plane_estimator, trivial) {
  point3d center(10.0, 11.0, 12.0);

  points3d points(6, 3);
  points <<
    center + vector3d(-3.0, 0.0, 0.0),
    center + vector3d(3.0, 0.0, 0.0),
    center + vector3d(0.0, -2.0, 0.0),
    center + vector3d(0.0, 2.0, 0.0),
    center + vector3d(0.0, 0.0, -1.0),
    center + vector3d(0.0, 0.0, 1.0);

  auto estimator = plane_estimator(points);

  EXPECT_DOUBLE_EQ(std::sqrt(28.0 / 6.0), estimator.point_error());
  EXPECT_DOUBLE_EQ(std::sqrt(10.0 / 6.0), estimator.line_error());
  EXPECT_DOUBLE_EQ(std::sqrt(2.0 / 6.0), estimator.plane_error());

  vector3d normal_expected(0.0, 0.0, 1.0);
  auto normal = estimator.plane_normal();
  EXPECT_DOUBLE_EQ(1.0, std::abs(normal_expected.dot(normal)));
}
