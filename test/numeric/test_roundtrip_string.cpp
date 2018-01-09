// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <limits>

#include <boost/lexical_cast.hpp>
#include <gtest/gtest.h>

#include <polatory/numeric/roundtrip_string.hpp>

using polatory::numeric::to_double;
using polatory::numeric::to_float;
using polatory::numeric::to_string;

TEST(roundtrip_string, denorm_min) {
  auto f = std::numeric_limits<float>::denorm_min();
  auto d = std::numeric_limits<double>::denorm_min();

  EXPECT_EQ(f, to_float(to_string(f)));
  EXPECT_EQ(d, to_double(to_string(d)));
}

TEST(roundtrip_string, infinity) {
  auto f = std::numeric_limits<float>::infinity();
  auto d = std::numeric_limits<double>::infinity();

  EXPECT_EQ(f, to_float(to_string(f)));
  EXPECT_EQ(d, to_double(to_string(d)));
}

TEST(roundtrip_string, negative_infinity) {
  auto f = -std::numeric_limits<float>::infinity();
  auto d = -std::numeric_limits<double>::infinity();

  EXPECT_EQ(f, to_float(to_string(f)));
  EXPECT_EQ(d, to_double(to_string(d)));
}

TEST(roundtrip_string, nan) {
  auto f = std::numeric_limits<float>::quiet_NaN();
  auto d = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(std::isnan(to_float(to_string(f))));
  EXPECT_TRUE(std::isnan(to_double(to_string(d))));
}
