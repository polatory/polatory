// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include "polatory/numeric/sum_accumulator.hpp"

using namespace polatory::numeric;

namespace {

template <typename Accumulator>
void test_sum_accumulator() {
  auto accum = Accumulator();
  for (int i = 0; i < 1000000; i++) {
    accum += 0.1;
  }
  ASSERT_DOUBLE_EQ(1e5, accum.get());
}

} // namespace

TEST(kahan_sum_accumulator, trivial) {
  test_sum_accumulator<kahan_sum_accumulator<double>>();
}

TEST(knuth_sum_accumulator, trivial) {
  test_sum_accumulator<knuth_sum_accumulator<double>>();
}
