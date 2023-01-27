#include <gtest/gtest.h>

#include <functional>
#include <polatory/common/fold.hpp>

using polatory::common::fold_left;
using polatory::common::fold_right;

TEST(fold_left, trivial) {
  EXPECT_EQ(42, fold_left(std::plus<>(), 42));
  EXPECT_EQ(42, fold_left(std::minus<>(), 42));

  EXPECT_EQ(15, fold_left(std::plus<>(), 1, 2, 3, 4, 5));
  EXPECT_EQ(-13, fold_left(std::minus<>(), 1, 2, 3, 4, 5));
}

TEST(fold_right, trivial) {
  EXPECT_EQ(42, fold_right(std::plus<>(), 42));
  EXPECT_EQ(42, fold_right(std::minus<>(), 42));

  EXPECT_EQ(15, fold_right(std::plus<>(), 1, 2, 3, 4, 5));
  EXPECT_EQ(3, fold_right(std::minus<>(), 1, 2, 3, 4, 5));
}
