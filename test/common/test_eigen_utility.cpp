#include <gtest/gtest.h>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/types.hpp>

using polatory::matrixd;
using polatory::common::concatenate_cols;
using polatory::common::concatenate_rows;

TEST(concatenate_cols, trivial) {
  matrixd a = matrixd::Random(3, 1);
  matrixd b = matrixd::Random(3, 2);
  matrixd c = matrixd::Random(3, 3);

  auto d = concatenate_cols(a, b, c);

  matrixd d_expected(3, 6);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}

TEST(concatenate_rows, trivial) {
  matrixd a = matrixd::Random(1, 3);
  matrixd b = matrixd::Random(2, 3);
  matrixd c = matrixd::Random(3, 3);

  auto d = concatenate_rows(a, b, c);

  matrixd d_expected(6, 3);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}
