// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>

using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;

TEST(bbox3d, from_points) {
  points3d points(7, 3);
  points <<
    point3d(0, 0, 0),
    point3d(-1, 0, 0),
    point3d(2, 0, 0),
    point3d(0, -3, 0),
    point3d(0, 4, 0),
    point3d(0, 0, -5),
    point3d(0, 0, 6);

  auto bbox = bbox3d::from_points(points);

  point3d min = points.colwise().minCoeff();
  point3d max = points.colwise().maxCoeff();
  auto bbox_expected = bbox3d(min, max);

  ASSERT_EQ(bbox_expected, bbox);
}
