// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "common/pi.hpp"
#include "geometry/affine_transform.hpp"

using polatory::geometry::affine_transform_point;
using polatory::geometry::affine_transform_vector;
using polatory::geometry::roll_pitch_yaw_matrix;
using polatory::geometry::scaling_matrix;
using polatory::common::pi;

TEST(affine_transform_point, trivial) {
  Eigen::Matrix4d m;
  m <<
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16;

  Eigen::Vector3d p(1, 2, 3);
  Eigen::Vector3d p2_expected(18, 46, 74);

  auto p2 = affine_transform_point(p, m);
  ASSERT_NEAR(0.0, (p2_expected - p2).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(affine_transform_vector, trivial) {
  Eigen::Matrix4d m;
  m <<
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16;

  Eigen::Vector3d v(1, 2, 3);
  Eigen::Vector3d v2_expected(14, 38, 62);

  auto v2 = affine_transform_vector(v, m);
  ASSERT_NEAR(0.0, (v2_expected - v2).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(roll_pitch_yaw_matrix, trivial) {
  auto m = roll_pitch_yaw_matrix(Eigen::Vector3d(pi / 3.0, pi / 5.0, pi / 7.0));

  Eigen::Matrix4d m_expected;
  m_expected <<
             0.4045084971874737, -0.7006292692220367, 0.5877852522924731, 0.0,
    0.9077771591538089, 0.22962157419990245, -0.35101931852905066, 0.0,
    0.11096623370094161, 0.6755683235405234, 0.7288991255358142, 0.0,
    0.0, 0.0, 0.0, 1.0;

  ASSERT_NEAR(0.0, (m_expected - m).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(scaling_matrix, trivial) {
  auto m = scaling_matrix(Eigen::Vector3d(3.0, 5.0, 7.0));

  Eigen::Matrix4d m_expected;
  m_expected <<
             3.0, 0.0, 0.0, 0.0,
    0.0, 5.0, 0.0, 0.0,
    0.0, 0.0, 7.0, 0.0,
    0.0, 0.0, 0.0, 1.0;

  ASSERT_EQ(0.0, (m_expected - m).lpNorm<Eigen::Infinity>());
}
