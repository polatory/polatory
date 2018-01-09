// Copyright (c) 2016, GSI and The Polatory Authors.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_direct_symmetric_evaluator.hpp>
#include <polatory/interpolation/rbf_operator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/biharmonic.hpp>

using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_symmetric_evaluator;
using polatory::interpolation::rbf_operator;
using polatory::point_cloud::random_points;
using polatory::polynomial::basis_base;
using polatory::rbf::biharmonic;

namespace {

void test_poly_degree(int poly_degree, size_t n_points) {
  size_t n_poly_basis = basis_base::basis_size(3, poly_degree);
  double absolute_tolerance = 5e-7;

  biharmonic rbf({ 1.0, 0.2 });

  auto points = random_points(sphere3d(), n_points);

  valuesd weights = valuesd::Random(n_points + n_poly_basis);

  rbf_direct_symmetric_evaluator direct_eval(rbf, 3, poly_degree, points);
  direct_eval.set_weights(weights);

  rbf_operator<> op(rbf, 3, poly_degree, points);

  valuesd direct_op_weights = direct_eval.evaluate() + rbf.nugget() * weights.head(n_points);
  valuesd op_weights = op(weights);

  EXPECT_EQ(n_points + n_poly_basis, op_weights.size());

  auto max_residual = (op_weights.head(n_points) - direct_op_weights).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);

  // TODO(mizuno): Test the polynomial part.
}

}  // namespace

TEST(rbf_operator, trivial) {
  test_poly_degree(-1, 1024);
  test_poly_degree(0, 1024);
  test_poly_degree(1, 1024);
  test_poly_degree(2, 1024);
}
