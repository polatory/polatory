// Copyright (c) 2016, GSI and The Polatory Authors.

#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/interpolation/rbf_direct_symmetric_evaluator.hpp"
#include "polatory/interpolation/rbf_operator.hpp"
#include "polatory/polynomial/basis_base.hpp"
#include "polatory/random_points/sphere_points.hpp"
#include "polatory/rbf/linear_variogram.hpp"

using namespace polatory::interpolation;
using polatory::random_points::sphere_points;
using polatory::polynomial::basis_base;
using polatory::rbf::linear_variogram;

namespace {

void test_poly_degree(int poly_degree, size_t n_points) {
  size_t n_polynomials = basis_base::dimension(poly_degree);
  double absolute_tolerance = 5e-7;

  linear_variogram rbf({ 1.0, 0.2 });

  auto points = sphere_points(n_points);

  Eigen::VectorXd weights = Eigen::VectorXd::Random(n_points + n_polynomials);

  rbf_direct_symmetric_evaluator direct_eval(rbf, poly_degree, points);
  direct_eval.set_weights(weights);

  rbf_operator<> op(rbf, poly_degree, points);

  Eigen::VectorXd direct_op_weights = direct_eval.evaluate() - rbf.nugget() * weights.head(n_points);
  Eigen::VectorXd op_weights = op(weights);

  ASSERT_EQ(n_points + n_polynomials, op_weights.size());

  auto max_residual = (op_weights.head(n_points) - direct_op_weights).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);

  // TODO: Test the polynomial part.
}

} // namespace

TEST(rbf_operator, trivial) {
  test_poly_degree(-1, 1024);
  test_poly_degree(0, 1024);
  test_poly_degree(1, 1024);
  test_poly_degree(2, 1024);
}
