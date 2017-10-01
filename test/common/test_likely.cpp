// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include "polatory/common/likely.hpp"

TEST(likely, trivial) {
  ASSERT_FALSE(LIKELY(false));
  ASSERT_TRUE(LIKELY(true));

  ASSERT_FALSE(UNLIKELY(false));
  ASSERT_TRUE(UNLIKELY(true));
}
