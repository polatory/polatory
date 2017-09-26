// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "geometry/affine_transform.hpp"
#include "pi.hpp"

using polatory::geometry::roll_pitch_yaw_matrix;
using polatory::geometry::scaling_matrix;
using polatory::pi;

TEST(roll_pitch_yaw_matrix, trivial)
{
   auto m = roll_pitch_yaw_matrix(Eigen::Vector3d(pi / 3.0, pi / 5.0, pi / 7.0));

   Eigen::Matrix4d m_expected;
   m_expected <<
      0.4045084971874737, -0.7006292692220367, 0.5877852522924731, 0.0,
      0.9077771591538089, 0.22962157419990245, -0.35101931852905066, 0.0,
      0.11096623370094161, 0.6755683235405234, 0.7288991255358142, 0.0,
      0.0, 0.0, 0.0, 1.0;

   ASSERT_NEAR(0.0, (m_expected - m).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(scaling_matrix, trivial)
{
   auto m = scaling_matrix(Eigen::Vector3d(3.0, 5.0, 7.0));

   Eigen::Matrix4d m_expected;
   m_expected <<
      3.0, 0.0, 0.0, 0.0,
      0.0, 5.0, 0.0, 0.0,
      0.0, 0.0, 7.0, 0.0,
      0.0, 0.0, 0.0, 1.0;

   ASSERT_EQ(0.0, (m_expected - m).lpNorm<Eigen::Infinity>());
}
