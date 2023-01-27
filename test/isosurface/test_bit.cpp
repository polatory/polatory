#include <gtest/gtest.h>

#include <polatory/isosurface/bit.hpp>

using polatory::isosurface::bit_count;
using polatory::isosurface::bit_peek;
using polatory::isosurface::bit_pop;

TEST(bit_count, trivial) {
  std::uint32_t decaf{0b1101'1110'1100'1010'1111};

  EXPECT_EQ(14, bit_count(decaf));
}

TEST(bit_peek, trivial) {
  EXPECT_EQ(0, bit_peek(std::uint8_t{0b1111}));
  EXPECT_EQ(1, bit_peek(std::uint8_t{0b1110}));
  EXPECT_EQ(2, bit_peek(std::uint8_t{0b1100}));
  EXPECT_EQ(3, bit_peek(std::uint8_t{0b1000}));
  EXPECT_EQ(-1, bit_peek(std::uint8_t{0}));
}

TEST(bit_pop, trivial) {
  std::uint8_t bit_set{0b1111};

  EXPECT_EQ(0, bit_pop(&bit_set));
  EXPECT_EQ(0b1110, bit_set);
  EXPECT_EQ(1, bit_pop(&bit_set));
  EXPECT_EQ(0b1100, bit_set);
  EXPECT_EQ(2, bit_pop(&bit_set));
  EXPECT_EQ(0b1000, bit_set);
  EXPECT_EQ(3, bit_pop(&bit_set));
  EXPECT_EQ(0, bit_set);
  EXPECT_EQ(-1, bit_pop(&bit_set));
  EXPECT_EQ(0, bit_set);

  std::uint32_t bit_set32{0b1000'0000'0000'0000'0000'0000'0000'0000};
  EXPECT_EQ(31, bit_pop(&bit_set32));
  EXPECT_EQ(0, bit_set32);

  std::uint64_t bit_set64{
      0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000};
  EXPECT_EQ(63, bit_pop(&bit_set64));
  EXPECT_EQ(0, bit_set64);
}
