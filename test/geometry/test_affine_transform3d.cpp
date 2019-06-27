// Copyright (c) 2016, GSI and The Polatory Authors.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/pi.hpp>
#include <polatory/geometry/affine_transformation3d.hpp>
#include <polatory/geometry/point3d.hpp>

using polatory::common::pi;
using polatory::geometry::affine_transformation3d;
using polatory::geometry::point3d;

TEST(affine_transformation3d, inverse) {
  Eigen::Matrix4d m;
  m <<
    1, 2, 9, 10,
    4, 3, 8, 11,
    5, 6, 7, 12,
    0, 0, 0, 1;
  affine_transformation3d t(m);
  auto ti = t.inverse();

  EXPECT_NEAR(0.0, (m.inverse() - ti.matrix()).lpNorm<Eigen::Infinity>(), 1e-12);
}

TEST(affine_transformation3d, transform_point) {
  Eigen::Matrix4d m;
  m <<
    1, 2, 9, 10,
    4, 3, 8, 11,
    5, 6, 7, 12,
    0, 0, 0, 1;
  affine_transformation3d affine(m);

  point3d p(1, 2, 3);
  point3d p2_expected(42, 45, 50);

  auto p2 = affine.transform_point(p);
  EXPECT_NEAR(0.0, (p2_expected - p2).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(affine_transformation3d, transform_vector) {
  Eigen::Matrix4d m;
  m <<
    1, 2, 9, 10,
    4, 3, 8, 11,
    5, 6, 7, 12,
    0, 0, 0, 1;
  affine_transformation3d t(m);

  point3d v(1, 2, 3);
  point3d v2_expected(32, 34, 38);

  auto v2 = t.transform_vector(v);
  EXPECT_NEAR(0.0, (v2_expected - v2).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(affine_transformation3d, roll_pitch_yaw) {
  auto m_actual = affine_transformation3d::roll_pitch_yaw(Eigen::Vector3d(pi<double>() / 3.0, pi<double>() / 5.0, pi<double>() / 7.0))
    .matrix();

  Eigen::Matrix4d m;
  m <<
    0.4045084971874737, -0.7006292692220367, 0.5877852522924731, 0.0,
    0.9077771591538089, 0.22962157419990245, -0.35101931852905066, 0.0,
    0.11096623370094161, 0.6755683235405234, 0.7288991255358142, 0.0,
    0.0, 0.0, 0.0, 1.0;

  EXPECT_NEAR(0.0, (m - m_actual).lpNorm<Eigen::Infinity>(), 1e-15);
}

TEST(affine_transformation3d, scaling) {
  auto m_actual = affine_transformation3d::scaling({ 3.0, 5.0, 7.0 })
    .matrix();

  Eigen::Matrix4d m;
  m <<
    3.0, 0.0, 0.0, 0.0,
    0.0, 5.0, 0.0, 0.0,
    0.0, 0.0, 7.0, 0.0,
    0.0, 0.0, 0.0, 1.0;

  EXPECT_EQ(0.0, (m - m_actual).lpNorm<Eigen::Infinity>());
}

TEST(affine_transformation3d, translation) {
  auto m_actual = affine_transformation3d::translation({ 3.0, 5.0, 7.0 })
    .matrix();

  Eigen::Matrix4d m;
  m <<
    0.0, 0.0, 0.0, 3.0,
    0.0, 0.0, 0.0, 5.0,
    0.0, 0.0, 0.0, 7.0,
    0.0, 0.0, 0.0, 1.0;

  EXPECT_EQ(0.0, (m - m_actual).lpNorm<Eigen::Infinity>());
}
