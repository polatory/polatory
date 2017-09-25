// Copyright (c) 2016, GSI and The Polatory Authors.

#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "point_cloud/distance_filter.hpp"

using namespace polatory::point_cloud;

TEST(distance_filter, trivial)
{
   std::vector<Eigen::Vector3d> points{
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(2, 0, 0),
      Eigen::Vector3d(2, 0, 0),
      Eigen::Vector3d(2, 0, 0)
   };

   Eigen::VectorXd values(6);
   values << 0, 1, 2, 3, 4, 5, 6, 7, 8;

   std::vector<Eigen::Vector3d> filtered_points_expected{
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(2, 0, 0)
   };

   Eigen::VectorXd filtered_values_expected(3);
   filtered_values_expected << 0, 3, 6;

   distance_filter filter(points, 0.1);
   auto filtered_points = filter.filtered_points();
   auto filtered_values = filter.filter_values(values);

   ASSERT_EQ(filtered_points, filtered_points_expected);
   ASSERT_EQ(filtered_values, filtered_values_expected);
}

TEST(distance_filter, filter_distance)
{
   std::vector<Eigen::Vector3d> points{
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(0, 0, 1),
      Eigen::Vector3d(2, 0, 0),
      Eigen::Vector3d(0, 2, 0),
      Eigen::Vector3d(0, 0, 2)
   };

   std::vector<Eigen::Vector3d> filtered_points_expected{
      Eigen::Vector3d(0, 0, 0),
      Eigen::Vector3d(2, 0, 0),
      Eigen::Vector3d(0, 2, 0),
      Eigen::Vector3d(0, 0, 2)
   };

   distance_filter filter(points, 1.5);
   auto filtered_points = filter.filtered_points();

   ASSERT_EQ(filtered_points, filtered_points_expected);
}
