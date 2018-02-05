// Copyright (c) 2016, GSI and The Polatory Authors.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/cov_exponential.hpp>

using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;
using polatory::point_cloud::random_points;
using polatory::rbf::cov_exponential;

namespace {

void test_poly_degree(int poly_degree, size_t n_points, size_t n_eval_points) {
  double absolute_tolerance = 1e-6;

  polatory::rbf::rbf rbf(cov_exponential({ 1.0, 0.2, 0.0 }), 3, poly_degree);

  auto points = random_points(sphere3d(), n_points);

  valuesd weights = valuesd::Random(n_points + rbf.poly_basis_size());

  rbf_direct_evaluator direct_eval(rbf, points);
  direct_eval.set_weights(weights);
  direct_eval.set_field_points(points.topRows(n_eval_points));

  rbf_evaluator<> eval(rbf, points);
  eval.set_weights(weights);
  eval.set_field_points(points);

  auto direct_values = direct_eval.evaluate();
  auto values = eval.evaluate();

  EXPECT_EQ(n_eval_points, direct_values.size());
  EXPECT_EQ(n_points, values.size());

  auto max_residual = (values.head(n_eval_points) - direct_values).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

}  // namespace

TEST(rbf_evaluator, trivial) {
  test_poly_degree(-1, 32768, 1024);
  test_poly_degree(0, 32768, 1024);
  test_poly_degree(1, 32768, 1024);
  test_poly_degree(2, 32768, 1024);
}
