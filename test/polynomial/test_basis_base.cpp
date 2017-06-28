// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include "polynomial/basis_base.hpp"

using namespace polatory::polynomial;

TEST(basis_base, trivial)
{
   ASSERT_EQ(0u, basis_base::dimension(-1));
   ASSERT_EQ(1u, basis_base::dimension(0));
   ASSERT_EQ(4u, basis_base::dimension(1));
   ASSERT_EQ(10u, basis_base::dimension(2));
}
