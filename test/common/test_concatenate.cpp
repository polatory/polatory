#include <gtest/gtest.h>

#include <jizai/common/concatenate.hpp>
#include <jizai/types.hpp>

using jizai::MatX;
using jizai::common::concatenate_cols;
using jizai::common::concatenate_rows;

TEST(concatenate_cols, trivial) {
  MatX a = MatX::Random(3, 1);
  MatX b = MatX::Random(3, 2);
  MatX c = MatX::Random(3, 3);

  auto d = concatenate_cols<MatX>(a, b, c);

  MatX d_expected(3, 6);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}

TEST(concatenate_rows, trivial) {
  MatX a = MatX::Random(1, 3);
  MatX b = MatX::Random(2, 3);
  MatX c = MatX::Random(3, 3);

  auto d = concatenate_rows<MatX>(a, b, c);

  MatX d_expected(6, 3);
  d_expected << a, b, c;

  EXPECT_EQ(d_expected, d);
}
