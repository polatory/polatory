// Copyright (c) 2016, GSI and The Polatory Authors.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/eigen_utility.hpp>

using namespace polatory::common;

TEST(concatenate_cols, trivial) {
  Eigen::MatrixXd a(3, 1);
  a <<
    0,
    1,
    2;

  Eigen::MatrixXd b(3, 2);
  b <<
    3, 6,
    4, 7,
    5, 8;

  Eigen::MatrixXd c(3, 3);
  c <<
    9, 12, 15,
    10, 13, 16,
    11, 14, 17;

  auto d = concatenate_cols(a, b, c);

  Eigen::MatrixXd d_expected(3, 6);
  d_expected <<
    0, 3, 6, 9, 12, 15,
    1, 4, 7, 10, 13, 16,
    2, 5, 8, 11, 14, 17;

  ASSERT_EQ(d_expected, d);
}

TEST(concatenate_rows, trivial) {
  Eigen::MatrixXd a(1, 3);
  a <<
    0, 1, 2;

  Eigen::MatrixXd b(2, 3);
  b <<
    3, 4, 5,
    6, 7, 8;

  Eigen::MatrixXd c(3, 3);
  c <<
    9, 10, 11,
    12, 13, 14,
    15, 16, 17;

  auto d = concatenate_rows(a, b, c);

  Eigen::MatrixXd d_expected(6, 3);
  d_expected <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11,
    12, 13, 14,
    15, 16, 17;

  ASSERT_EQ(d_expected, d);
}
