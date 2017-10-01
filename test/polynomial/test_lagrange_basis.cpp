// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/polynomial/lagrange_basis.hpp"

using namespace polatory::polynomial;

namespace {

void test_degree(int degree) {
  size_t n_points = lagrange_basis<>::dimension(degree);

  std::vector<Eigen::Vector3d> points;
  points.reserve(n_points);

  for (size_t i = 0; i < n_points; i++) {
    points.push_back(Eigen::Vector3d::Random());
  }

  lagrange_basis<> basis(degree, points);
  auto pt = basis.evaluate_points(points);

  ASSERT_EQ(basis.dimension(), pt.rows());
  ASSERT_EQ(n_points, pt.cols());

  Eigen::MatrixXd diff = Eigen::MatrixXd::Identity(n_points, n_points) - pt;

  EXPECT_LT(diff.lpNorm<Eigen::Infinity>(), 1e-12);
}

} // namespace

TEST(lagrange_basis, trivial) {
  test_degree(0);
  test_degree(1);
  test_degree(2);
}
