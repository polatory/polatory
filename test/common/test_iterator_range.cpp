#include <vector>

#include <gtest/gtest.h>

#include <polatory/common/iterator_range.hpp>

using polatory::common::make_range;

TEST(iterator_range, trivial) {
  std::vector<int> v{ 0, 1, 2, 3, 4 };
  auto range = make_range(v.begin(), v.end());

  EXPECT_EQ(v.begin(), range.begin());
  EXPECT_EQ(v.end(), range.end());
}
