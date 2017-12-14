// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/common/likely.hpp>

TEST(likely, trivial) {
  ASSERT_FALSE(POLATORY_LIKELY(false));
  ASSERT_TRUE(POLATORY_LIKELY(true));

  ASSERT_FALSE(POLATORY_UNLIKELY(false));
  ASSERT_TRUE(POLATORY_UNLIKELY(true));
}
