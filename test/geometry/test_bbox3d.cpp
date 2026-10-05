#include <gtest/gtest.h>

#include <jizai/geometry/bbox3d.hpp>
#include <jizai/geometry/point3d.hpp>

using jizai::geometry::Bbox3;
using jizai::geometry::Point3;
using jizai::geometry::Points3;

TEST(bbox3d, from_points) {
  Points3 points(7, 3);
  points << Point3(0, 0, 0), Point3(-1, 0, 0), Point3(2, 0, 0), Point3(0, -3, 0), Point3(0, 4, 0),
      Point3(0, 0, -5), Point3(0, 0, 6);

  auto bbox = Bbox3::from_points(points);

  Point3 min = points.colwise().minCoeff();
  Point3 max = points.colwise().maxCoeff();
  auto bbox_expected = Bbox3(min, max);

  EXPECT_EQ(bbox_expected, bbox);
}
