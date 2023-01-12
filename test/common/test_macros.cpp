#include <gtest/gtest.h>

#include <polatory/common/macros.hpp>

TEST(likely, trivial) {
  EXPECT_FALSE(POLATORY_LIKELY(false));
  EXPECT_TRUE(POLATORY_LIKELY(true));

  EXPECT_FALSE(POLATORY_UNLIKELY(false));
  EXPECT_TRUE(POLATORY_UNLIKELY(true));
}
