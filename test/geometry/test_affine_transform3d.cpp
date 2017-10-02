// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/common/pi.hpp"
#include "polatory/geometry/affine_transform3d.hpp"

using namespace polatory::geometry;
using polatory::common::pi;

TEST(affine_transform3d, transform_point) {
  Eigen::Matrix4d m;
  m <<
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    0, 0, 0, 1;
  affine_transform3d affine(m);

  Eigen::Vector3d p(1, 2, 3);
  Eigen::Vector3d p2_expected(18, 46, 74);

  auto p2 = affine.transform_point(p);
  ASSERT_NEAR(0.0, (p2_expected - p2).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(affine_transform3d, transform_vector) {
  Eigen::Matrix4d m;
  m <<
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    0, 0, 0, 1;
  affine_transform3d affine(m);

  Eigen::Vector3d v(1, 2, 3);
  Eigen::Vector3d v2_expected(14, 38, 62);

  auto v2 = affine.transform_vector(v);
  ASSERT_NEAR(0.0, (v2_expected - v2).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(affine_transform3d, roll_pitch_yaw) {
  auto m_actual = affine_transform3d::roll_pitch_yaw(Eigen::Vector3d(pi / 3.0, pi / 5.0, pi / 7.0))
    .matrix();

  Eigen::Matrix4d m;
  m <<
    0.4045084971874737, -0.7006292692220367, 0.5877852522924731, 0.0,
    0.9077771591538089, 0.22962157419990245, -0.35101931852905066, 0.0,
    0.11096623370094161, 0.6755683235405234, 0.7288991255358142, 0.0,
    0.0, 0.0, 0.0, 1.0;

  ASSERT_NEAR(0.0, (m - m_actual).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(affine_transform3d, scaling) {
  auto m_actual = affine_transform3d::scaling(Eigen::Vector3d(3.0, 5.0, 7.0))
    .matrix();

  Eigen::Matrix4d m;
  m <<
    3.0, 0.0, 0.0, 0.0,
    0.0, 5.0, 0.0, 0.0,
    0.0, 0.0, 7.0, 0.0,
    0.0, 0.0, 0.0, 1.0;

  ASSERT_EQ(0.0, (m - m_actual).lpNorm<Eigen::Infinity>());
}

TEST(affine_transform3d, translation) {
  auto m_actual = affine_transform3d::translation(Eigen::Vector3d(3.0, 5.0, 7.0))
    .matrix();

  Eigen::Matrix4d m;
  m <<
    0.0, 0.0, 0.0, 3.0,
    0.0, 0.0, 0.0, 5.0,
    0.0, 0.0, 0.0, 7.0,
    0.0, 0.0, 0.0, 1.0;

  ASSERT_EQ(0.0, (m - m_actual).lpNorm<Eigen::Infinity>());
}
