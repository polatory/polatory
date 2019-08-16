// Copyright (c) 2016, GSI and The Polatory Authors.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_direct_symmetric_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/types.hpp>

#include "random_transformation.hpp"

using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_direct_symmetric_evaluator;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::model;
using polatory::point_cloud::random_points;
using polatory::rbf::cov_exponential;
using polatory::index_t;

namespace {

template <class Evaluator>
void test_poly_degree(int poly_degree, index_t n_points, index_t n_eval_points) {
  auto absolute_tolerance = 2e-6;

  cov_exponential rbf({ 1.0, 0.2 });
  rbf.set_transformation(random_transformation());

  model model(rbf, 3, poly_degree);

  auto points = random_points(sphere3d(), n_points);

  valuesd weights = valuesd::Random(n_points + model.poly_basis_size());

  rbf_direct_evaluator direct_eval(model, points);
  direct_eval.set_weights(weights);
  direct_eval.set_field_points(points.topRows(n_eval_points));

  Evaluator eval(model, points);
  eval.set_weights(weights);

  auto direct_values = direct_eval.evaluate();
  auto values = eval.evaluate();

  EXPECT_EQ(n_eval_points, direct_values.size());
  EXPECT_EQ(n_points, values.size());

  auto max_residual = (values.head(n_eval_points) - direct_values).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

}  // namespace

TEST(rbf_direct_symmetric_evaluator, trivial) {
  test_poly_degree<rbf_direct_symmetric_evaluator>(-1, 1024, 1024);
  test_poly_degree<rbf_direct_symmetric_evaluator>(0, 1024, 1024);
  test_poly_degree<rbf_direct_symmetric_evaluator>(1, 1024, 1024);
  test_poly_degree<rbf_direct_symmetric_evaluator>(2, 1024, 1024);
}

TEST(rbf_symmetric_evaluator, trivial) {
  test_poly_degree<rbf_symmetric_evaluator<>>(-1, 32768, 1024);
  test_poly_degree<rbf_symmetric_evaluator<>>(0, 32768, 1024);
  test_poly_degree<rbf_symmetric_evaluator<>>(1, 32768, 1024);
  test_poly_degree<rbf_symmetric_evaluator<>>(2, 32768, 1024);
}
