#include <gtest/gtest.h>

#include <cmath>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/plane_estimator.hpp>

using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::geometry::Vector3;
using polatory::point_cloud::PlaneEstimator;

TEST(plane_estimator, trivial) {
  Point3 center(10.0, 11.0, 12.0);

  Points3 points(6, 3);
  points << center + Vector3(-3.0, 0.0, 0.0), center + Vector3(3.0, 0.0, 0.0),
      center + Vector3(0.0, -2.0, 0.0), center + Vector3(0.0, 2.0, 0.0),
      center + Vector3(0.0, 0.0, -1.0), center + Vector3(0.0, 0.0, 1.0);

  auto estimator = PlaneEstimator(points);

  EXPECT_DOUBLE_EQ(std::sqrt(28.0 / 6.0), estimator.point_error());
  EXPECT_DOUBLE_EQ(std::sqrt(10.0 / 6.0), estimator.line_error());
  EXPECT_DOUBLE_EQ(std::sqrt(2.0 / 6.0), estimator.plane_error());

  Vector3 normal_expected(0.0, 0.0, 1.0);
  auto normal = estimator.plane_normal();
  EXPECT_DOUBLE_EQ(1.0, std::abs(normal_expected.dot(normal)));
}
