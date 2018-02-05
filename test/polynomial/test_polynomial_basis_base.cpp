// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <polatory/polynomial/polynomial_basis_base.hpp>

using polatory::polynomial::polynomial_basis_base;

TEST(polynomial_basis_base, trivial) {
  EXPECT_EQ(0u, polynomial_basis_base::basis_size(-1, -1));

  EXPECT_EQ(1u, polynomial_basis_base::basis_size(1, 0));
  EXPECT_EQ(2u, polynomial_basis_base::basis_size(1, 1));
  EXPECT_EQ(3u, polynomial_basis_base::basis_size(1, 2));

  EXPECT_EQ(1u, polynomial_basis_base::basis_size(2, 0));
  EXPECT_EQ(3u, polynomial_basis_base::basis_size(2, 1));
  EXPECT_EQ(6u, polynomial_basis_base::basis_size(2, 2));

  EXPECT_EQ(1u, polynomial_basis_base::basis_size(3, 0));
  EXPECT_EQ(4u, polynomial_basis_base::basis_size(3, 1));
  EXPECT_EQ(10u, polynomial_basis_base::basis_size(3, 2));
}
