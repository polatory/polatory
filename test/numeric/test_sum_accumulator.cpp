// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/numeric/sum_accumulator.hpp>

using polatory::numeric::kahan_sum_accumulator;
using polatory::numeric::knuth_sum_accumulator;

namespace {

template <class Accumulator>
void test_sum_accumulator() {
  Accumulator accum;
  for (size_t i = 0; i < 1000000; i++) {
    accum += 0.1;
  }
  EXPECT_DOUBLE_EQ(1e5, accum.get());
}

}  // namespace

TEST(kahan_sum_accumulator, trivial) {
  test_sum_accumulator<kahan_sum_accumulator<double>>();
}

TEST(knuth_sum_accumulator, trivial) {
  test_sum_accumulator<knuth_sum_accumulator<double>>();
}
