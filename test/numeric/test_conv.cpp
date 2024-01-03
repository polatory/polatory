#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <polatory/numeric/conv.hpp>

using polatory::numeric::to_double;
using polatory::numeric::to_float;
using polatory::numeric::to_string;

TEST(conv, denorm_min) {
  auto f = std::numeric_limits<float>::denorm_min();
  auto d = std::numeric_limits<double>::denorm_min();

  EXPECT_EQ(f, to_float(to_string(f)));
  EXPECT_EQ(d, to_double(to_string(d)));
}

TEST(conv, infinity) {
  auto f = std::numeric_limits<float>::infinity();
  auto d = std::numeric_limits<double>::infinity();

  EXPECT_EQ(f, to_float(to_string(f)));
  EXPECT_EQ(d, to_double(to_string(d)));
}

TEST(conv, negative_infinity) {
  auto f = -std::numeric_limits<float>::infinity();
  auto d = -std::numeric_limits<double>::infinity();

  EXPECT_EQ(f, to_float(to_string(f)));
  EXPECT_EQ(d, to_double(to_string(d)));
}

TEST(conv, nan) {
  auto f = std::numeric_limits<float>::quiet_NaN();
  auto d = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(std::isnan(to_float(to_string(f))));
  EXPECT_TRUE(std::isnan(to_double(to_string(d))));
}
