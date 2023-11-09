#include <gtest/gtest.h>

#include <polatory/polynomial/polynomial_basis_base.hpp>

using polatory::polynomial::polynomial_basis_base;

TEST(polynomial_basis_base, trivial) {
  EXPECT_EQ(0, polynomial_basis_base<1>::basis_size(-1));
  EXPECT_EQ(1, polynomial_basis_base<1>::basis_size(0));
  EXPECT_EQ(2, polynomial_basis_base<1>::basis_size(1));
  EXPECT_EQ(3, polynomial_basis_base<1>::basis_size(2));

  EXPECT_EQ(0, polynomial_basis_base<2>::basis_size(-1));
  EXPECT_EQ(1, polynomial_basis_base<2>::basis_size(0));
  EXPECT_EQ(3, polynomial_basis_base<2>::basis_size(1));
  EXPECT_EQ(6, polynomial_basis_base<2>::basis_size(2));

  EXPECT_EQ(0, polynomial_basis_base<3>::basis_size(-1));
  EXPECT_EQ(1, polynomial_basis_base<3>::basis_size(0));
  EXPECT_EQ(4, polynomial_basis_base<3>::basis_size(1));
  EXPECT_EQ(10, polynomial_basis_base<3>::basis_size(2));
}
