// Copyright (c) 2016, GSI and The Polatory Authors.

#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>

using namespace polatory::point_cloud;
using polatory::geometry::point3d;
using polatory::geometry::points3d;

TEST(distance_filter, trivial) {
  points3d points(9);
  points <<
    point3d(0, 0, 0),
    point3d(0, 0, 0),
    point3d(0, 0, 0),
    point3d(1, 0, 0),
    point3d(1, 0, 0),
    point3d(1, 0, 0),
    point3d(2, 0, 0),
    point3d(2, 0, 0),
    point3d(2, 0, 0);

  Eigen::VectorXd values(9);
  values << 0, 1, 2, 3, 4, 5, 6, 7, 8;

  points3d filtered_points_expected(3);
  filtered_points_expected <<
    point3d(0, 0, 0),
    point3d(1, 0, 0),
    point3d(2, 0, 0);

  Eigen::VectorXd filtered_values_expected(3);
  filtered_values_expected << 0, 3, 6;

  distance_filter filter(points, 0.1);
  auto filtered_points = filter.filtered(points);
  auto filtered_values = filter.filtered(values);

  ASSERT_EQ(filtered_points_expected, filtered_points);
  ASSERT_EQ(filtered_values_expected, filtered_values);
}

TEST(distance_filter, filter_distance) {
  points3d points(7);
  points <<
    point3d(0, 0, 0),
    point3d(1, 0, 0),
    point3d(0, 1, 0),
    point3d(0, 0, 1),
    point3d(2, 0, 0),
    point3d(0, 2, 0),
    point3d(0, 0, 2);

  points3d filtered_points_expected(4);
  filtered_points_expected <<
    point3d(0, 0, 0),
    point3d(2, 0, 0),
    point3d(0, 2, 0),
    point3d(0, 0, 2);

  distance_filter filter(points, 1.5);
  auto filtered_points = filter.filtered(points);

  ASSERT_EQ(filtered_points_expected, filtered_points);
}
