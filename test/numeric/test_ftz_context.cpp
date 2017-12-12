// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/numeric/ftz_context.hpp>

using polatory::numeric::ftz_context;

TEST(ftz_context, trivial) {
  ASSERT_FALSE(ftz_context::daz_is_active());
  ASSERT_FALSE(ftz_context::ftz_is_active());

  {
    ASSERT_FALSE(ftz_context::daz_is_active());
    ASSERT_FALSE(ftz_context::ftz_is_active());

    ftz_context ftz;

    ASSERT_TRUE(ftz_context::daz_is_active());
    ASSERT_TRUE(ftz_context::ftz_is_active());
  }

  ASSERT_FALSE(ftz_context::daz_is_active());
  ASSERT_FALSE(ftz_context::ftz_is_active());
}
