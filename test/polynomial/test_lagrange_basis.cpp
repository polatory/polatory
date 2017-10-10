// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/polynomial/lagrange_basis.hpp"

using namespace polatory::polynomial;

namespace {

void test_degree(int dimension, int degree) {
  size_t n_points = lagrange_basis<>::basis_size(dimension, degree);

  std::vector<Eigen::Vector3d> points;
  points.reserve(n_points);

  for (size_t i = 0; i < n_points; i++) {
    points.push_back(Eigen::Vector3d::Random());
  }

  lagrange_basis<> basis(dimension, degree, points);
  auto pt = basis.evaluate_points(points);

  ASSERT_EQ(basis.basis_size(), pt.rows());
  ASSERT_EQ(n_points, pt.cols());

  Eigen::MatrixXd diff = Eigen::MatrixXd::Identity(n_points, n_points) - pt;

  EXPECT_LT(diff.lpNorm<Eigen::Infinity>(), 1e-12);
}

} // namespace

TEST(lagrange_basis, trivial) {
  for (int dim = 1; dim <= 3; dim++) {
    for (int deg = 0; deg <= 2; deg++) {
      test_degree(dim, deg);
    }
  }
}
