// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/numeric/scoped_ftz.hpp>

using polatory::numeric::scoped_ftz;

TEST(scoped_ftz, trivial) {
#ifdef POLATORY_ENABLE_FTZ
  EXPECT_TRUE(scoped_ftz::daz_is_active());
  EXPECT_TRUE(scoped_ftz::ftz_is_active());
#else
  EXPECT_FALSE(scoped_ftz::daz_is_active());
  EXPECT_FALSE(scoped_ftz::ftz_is_active());

  {
    EXPECT_FALSE(scoped_ftz::daz_is_active());
    EXPECT_FALSE(scoped_ftz::ftz_is_active());

    scoped_ftz ftz;

    EXPECT_TRUE(scoped_ftz::daz_is_active());
    EXPECT_TRUE(scoped_ftz::ftz_is_active());
  }

  EXPECT_FALSE(scoped_ftz::daz_is_active());
  EXPECT_FALSE(scoped_ftz::ftz_is_active());
#endif
}
