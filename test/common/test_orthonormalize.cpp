#include <gtest/gtest.h>

#include <cmath>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/types.hpp>

using polatory::index_t;
using polatory::matrixd;
using polatory::common::orthonormalize_cols;

TEST(orthonormalize_cols, trivial) {
  const index_t rows = 100;
  const index_t cols = 10;

  matrixd m = matrixd::Random(rows, cols);
  orthonormalize_cols(m);

  for (index_t i = 0; i < cols; i++) {
    for (index_t j = i; j < cols; j++) {
      auto dot = std::abs(m.col(i).dot(m.col(j)));

      if (i == j) {
        EXPECT_LT(std::abs(dot - 1.0), 1e-15);
      } else {
        EXPECT_LT(std::abs(dot), 1e-15);
      }
    }
  }
}
