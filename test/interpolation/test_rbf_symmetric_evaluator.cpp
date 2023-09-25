#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/multiquadric1.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::point_cloud::random_points;
using polatory::rbf::multiquadric1;

namespace {

void test_poly_degree(int poly_degree) {
  const int dim = 3;
  const index_t n_points = 32768;
  const index_t n_grad_points = 4096;
  const index_t n_eval_points = 1024;
  const index_t n_eval_grad_points = 1024;

  auto absolute_tolerance = 2e-6;

  multiquadric1 rbf({1.0, 0.001});
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, dim, poly_degree);

  auto points = random_points(sphere3d(), n_points);
  auto grad_points = random_points(sphere3d(), n_grad_points);

  valuesd weights = valuesd::Random(n_points + dim * n_grad_points + model.poly_basis_size());

  rbf_direct_evaluator direct_eval(model, points, grad_points);
  direct_eval.set_weights(weights);
  direct_eval.set_field_points(points.topRows(n_eval_points),
                               grad_points.topRows(n_eval_grad_points));

  rbf_symmetric_evaluator<> eval(model, points, grad_points);
  eval.set_weights(weights);

  auto direct_values = direct_eval.evaluate();
  auto values_full = eval.evaluate();

  EXPECT_EQ(n_eval_points + dim * n_eval_grad_points, direct_values.rows());
  EXPECT_EQ(n_points + dim * n_grad_points, values_full.rows());

  valuesd values(direct_values.size());
  values << values_full.head(n_eval_points),
      values_full.segment(n_points, dim * n_eval_grad_points);

  auto max_residual = (values - direct_values).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

}  // namespace

TEST(rbf_symmetric_evaluator, trivial) {
  test_poly_degree(0);
  test_poly_degree(1);
  test_poly_degree(2);
}
