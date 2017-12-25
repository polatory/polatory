// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/isosurface/bit.hpp>

using polatory::isosurface::bit_count;
using polatory::isosurface::bit_peek;
using polatory::isosurface::bit_pop;

TEST(bit_count, trivial) {
  int decaf = 0b1101'1110'1100'1010'1111;

  ASSERT_EQ(14, bit_count(decaf));
}

TEST(bit_peek, trivial) {
  ASSERT_EQ(0, bit_peek(0b1111));
  ASSERT_EQ(1, bit_peek(0b1110));
  ASSERT_EQ(2, bit_peek(0b1100));
  ASSERT_EQ(3, bit_peek(0b1000));
  ASSERT_EQ(-1, bit_peek(0));
}

TEST(bit_pop, trivial) {
  int bit_set = 0b1111;

  ASSERT_EQ(0, bit_pop(&bit_set));
  ASSERT_EQ(0b1110, bit_set);
  ASSERT_EQ(1, bit_pop(&bit_set));
  ASSERT_EQ(0b1100, bit_set);
  ASSERT_EQ(2, bit_pop(&bit_set));
  ASSERT_EQ(0b1000, bit_set);
  ASSERT_EQ(3, bit_pop(&bit_set));
  ASSERT_EQ(0, bit_set);
  ASSERT_EQ(-1, bit_pop(&bit_set));
  ASSERT_EQ(0, bit_set);
}
