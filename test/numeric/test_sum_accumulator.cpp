// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/numeric/sum_accumulator.hpp>
#include <polatory/types.hpp>

using polatory::numeric::kahan_sum_accumulator;
using polatory::index_t;

namespace {

template <class Accumulator>
void test_sum_accumulator() {
  const auto n = index_t{ 1000000 };

  Accumulator accum;
  for (index_t i = 0; i < n; i++) {
    accum += 0.1;
  }
  EXPECT_DOUBLE_EQ(1e5, accum.get());
}

}  // namespace

TEST(kahan_sum_accumulator, trivial) {
  test_sum_accumulator<kahan_sum_accumulator<double>>();
}
