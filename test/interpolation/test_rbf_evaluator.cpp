// Copyright (c) 2016, GSI and The Polatory Authors.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/biharmonic.hpp>

using namespace polatory::interpolation;
using polatory::geometry::sphere3d;
using polatory::point_cloud::random_points;
using polatory::polynomial::basis_base;
using polatory::rbf::biharmonic;

namespace {

void test_poly_degree(int poly_degree, size_t n_points, size_t n_eval_points) {
  size_t n_poly_basis = basis_base::basis_size(3, poly_degree);
  double absolute_tolerance = 5e-7;

  biharmonic rbf({ 1.0, 0.0 });

  auto points = random_points(sphere3d(), n_points);

  Eigen::VectorXd weights = Eigen::VectorXd::Random(n_points + n_poly_basis);

  rbf_direct_evaluator direct_eval(rbf, 3, poly_degree, points);
  direct_eval.set_weights(weights);
  direct_eval.set_field_points(points.topRows(n_eval_points));

  rbf_evaluator<> eval(rbf, 3, poly_degree, points);
  eval.set_weights(weights);
  eval.set_field_points(points);

  auto direct_values = direct_eval.evaluate();
  auto values = eval.evaluate();

  ASSERT_EQ(n_eval_points, direct_values.size());
  ASSERT_EQ(n_points, values.size());

  auto max_residual = (values.head(n_eval_points) - direct_values).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

} // namespace

TEST(rbf_evaluator, trivial) {
  test_poly_degree(-1, 32768, 1024);
  test_poly_degree(0, 32768, 1024);
  test_poly_degree(1, 32768, 1024);
  test_poly_degree(2, 32768, 1024);
}
