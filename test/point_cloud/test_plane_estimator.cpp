// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/point_cloud/plane_estimator.hpp>

using namespace polatory::point_cloud;

TEST(plane_estimator, trivial) {
  std::vector<Eigen::Vector3d> points;

  Eigen::Vector3d center(10.0, 11.0, 12.0);

  points.push_back(center + Eigen::Vector3d(-3.0, 0.0, 0.0));
  points.push_back(center + Eigen::Vector3d(3.0, 0.0, 0.0));
  points.push_back(center + Eigen::Vector3d(0.0, -2.0, 0.0));
  points.push_back(center + Eigen::Vector3d(0.0, 2.0, 0.0));
  points.push_back(center + Eigen::Vector3d(0.0, 0.0, -1.0));
  points.push_back(center + Eigen::Vector3d(0.0, 0.0, 1.0));

  auto estimator = plane_estimator(points);

  ASSERT_DOUBLE_EQ(std::sqrt(28.0 / 6.0), estimator.point_error());
  ASSERT_DOUBLE_EQ(std::sqrt(10.0 / 6.0), estimator.line_error());
  ASSERT_DOUBLE_EQ(std::sqrt(2.0 / 6.0), estimator.plane_error());

  Eigen::Vector3d normal_expected(0.0, 0.0, 1.0);
  auto normal = estimator.plane_normal();
  ASSERT_DOUBLE_EQ(1.0, std::abs(normal_expected.dot(normal)));
}
