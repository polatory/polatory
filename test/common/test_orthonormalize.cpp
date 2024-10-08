#include <gtest/gtest.h>

#include <cmath>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/types.hpp>

using polatory::Index;
using polatory::MatX;
using polatory::common::orthonormalize_cols;

TEST(orthonormalize_cols, trivial) {
  const Index rows = 100;
  const Index cols = 10;

  MatX m = MatX::Random(rows, cols);
  orthonormalize_cols(m);

  for (Index i = 0; i < cols; i++) {
    for (Index j = i; j < cols; j++) {
      auto dot = std::abs(m.col(i).dot(m.col(j)));

      if (i == j) {
        EXPECT_LT(std::abs(dot - 1.0), 1e-15);
      } else {
        EXPECT_LT(std::abs(dot), 1e-15);
      }
    }
  }
}
