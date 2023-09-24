#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_evaluator;
using polatory::point_cloud::random_points;
using polatory::rbf::cov_exponential;

namespace {

void test_poly_degree(int poly_degree) {
  const int dim = 3;
  const index_t n_points = 1024;
  const index_t n_grad_points = 1024;
  const index_t n_eval_points = 1024;

  auto absolute_tolerance = n_grad_points > 0 ? 5e-5 : 2e-6;

  cov_exponential rbf({1.0, 0.2});
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, dim, poly_degree);

  auto points = random_points(sphere3d(), n_points);
  auto grad_points = random_points(sphere3d(), n_grad_points);
  auto eval_points = random_points(sphere3d(), n_eval_points);

  valuesd weights = valuesd::Random(n_points + dim * n_grad_points + model.poly_basis_size());

  rbf_direct_evaluator direct_eval(model, points, grad_points);
  direct_eval.set_weights(weights);
  direct_eval.set_field_points(eval_points);

  bbox3d bbox{-point3d::Ones(), point3d::Ones()};
  rbf_evaluator<> eval(model, points, grad_points, bbox);
  eval.set_weights(weights);
  eval.set_field_points(eval_points);

  auto direct_values = direct_eval.evaluate();
  auto values = eval.evaluate();

  EXPECT_EQ(n_eval_points, direct_values.rows());
  EXPECT_EQ(n_eval_points, values.rows());

  auto max_residual = (values - direct_values).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

}  // namespace

TEST(rbf_evaluator, trivial) {
  test_poly_degree(-1);
  // test_poly_degree(0);
  // test_poly_degree(1);
  // test_poly_degree(2);
}
