// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/polynomial/basis_base.hpp>

using polatory::polynomial::basis_base;

TEST(basis_base, trivial) {
  ASSERT_EQ(0u, basis_base::basis_size(-1, -1));

  ASSERT_EQ(1u, basis_base::basis_size(1, 0));
  ASSERT_EQ(2u, basis_base::basis_size(1, 1));
  ASSERT_EQ(3u, basis_base::basis_size(1, 2));

  ASSERT_EQ(1u, basis_base::basis_size(2, 0));
  ASSERT_EQ(3u, basis_base::basis_size(2, 1));
  ASSERT_EQ(6u, basis_base::basis_size(2, 2));

  ASSERT_EQ(1u, basis_base::basis_size(3, 0));
  ASSERT_EQ(4u, basis_base::basis_size(3, 1));
  ASSERT_EQ(10u, basis_base::basis_size(3, 2));
}
