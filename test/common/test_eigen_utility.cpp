#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/common/eigen_utility.hpp>

using polatory::common::concatenate_cols;
using polatory::common::concatenate_rows;

TEST(concatenate_cols, trivial) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 1);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(3, 2);
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(3, 3);

  auto d = concatenate_cols(a, b, c);

  Eigen::MatrixXd d_expected(3, 6);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}

TEST(concatenate_rows, trivial) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(3, 3);

  auto d = concatenate_rows(a, b, c);

  Eigen::MatrixXd d_expected(6, 3);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}
