// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <polatory/common/quasi_random_sequence.hpp>

using polatory::common::quasi_random_sequence;

namespace {

void test_n(size_t n) {
  auto seq = quasi_random_sequence(n);
  ASSERT_EQ(n, seq.size());

  std::sort(seq.begin(), seq.end());

  std::vector<size_t> expected(n);
  std::iota(expected.begin(), expected.end(), 0);

  ASSERT_EQ(expected, seq);
}

}  // namespace

TEST(quasi_random_sequence, trivial) {
  for (size_t n = 0; n < 16; n++) {
    test_n(n);
  }

  test_n(1023);
}
