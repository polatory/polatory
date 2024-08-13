#include <gtest/gtest.h>

#include <polatory/polynomial/polynomial_basis_base.hpp>

using polatory::polynomial::PolynomialBasisBase;

TEST(polynomial_basis_base, trivial) {
  EXPECT_EQ(0, PolynomialBasisBase<1>::basis_size(-1));
  EXPECT_EQ(1, PolynomialBasisBase<1>::basis_size(0));
  EXPECT_EQ(2, PolynomialBasisBase<1>::basis_size(1));
  EXPECT_EQ(3, PolynomialBasisBase<1>::basis_size(2));

  EXPECT_EQ(0, PolynomialBasisBase<2>::basis_size(-1));
  EXPECT_EQ(1, PolynomialBasisBase<2>::basis_size(0));
  EXPECT_EQ(3, PolynomialBasisBase<2>::basis_size(1));
  EXPECT_EQ(6, PolynomialBasisBase<2>::basis_size(2));

  EXPECT_EQ(0, PolynomialBasisBase<3>::basis_size(-1));
  EXPECT_EQ(1, PolynomialBasisBase<3>::basis_size(0));
  EXPECT_EQ(4, PolynomialBasisBase<3>::basis_size(1));
  EXPECT_EQ(10, PolynomialBasisBase<3>::basis_size(2));
}
