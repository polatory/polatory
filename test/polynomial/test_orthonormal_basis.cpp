// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/polynomial/orthonormal_basis.hpp"

using namespace polatory::polynomial;

namespace {

void test_degree(int degree) {
  int n_points = 100;

  std::vector<Eigen::Vector3d> points;
  points.reserve(n_points);

  for (int i = 0; i < n_points; i++) {
    points.push_back(Eigen::Vector3d::Random());
  }

  orthonormal_basis<> basis(degree, points);
  auto pt = basis.evaluate_points(points);
  auto dim = basis.dimension();

  ASSERT_EQ(basis.dimension(), pt.rows());
  ASSERT_EQ(n_points, pt.cols());

  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      auto dot = std::abs(pt.row(i).dot(pt.row(j)));

      if (i == j) {
        EXPECT_LT(std::abs(dot - 1.0), 1e-12);
      } else {
        EXPECT_LT(std::abs(dot), 1e-12);
      }
    }
  }
}

} // namespace

TEST(orthonormal_basis, trivial) {
  test_degree(0);
  test_degree(1);
  test_degree(2);
}
