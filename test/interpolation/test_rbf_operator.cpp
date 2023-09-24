#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/interpolation/rbf_direct_operator.hpp>
#include <polatory/interpolation/rbf_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/multiquadric1.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_operator;
using polatory::interpolation::rbf_operator;
using polatory::point_cloud::random_points;
using polatory::rbf::multiquadric1;

namespace {

void test_poly_degree(int poly_degree) {
  const int dim = 3;
  const index_t n_points = 1024;
  const index_t n_grad_points = 1024;

  auto absolute_tolerance = 2e-6;

  multiquadric1 rbf({1.0, 0.001});
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, dim, poly_degree);
  // model.set_nugget(0.01);

  auto points = random_points(sphere3d(), n_points);
  auto grad_points = random_points(sphere3d(), n_grad_points);
  valuesd weights = valuesd::Random(n_points + dim * n_grad_points + model.poly_basis_size());

  rbf_direct_operator direct_op(model, points, grad_points);
  rbf_operator<> op(model, points, grad_points);

  valuesd direct_op_weights = direct_op(weights);
  valuesd op_weights = op(weights);

  EXPECT_EQ(n_points + dim * n_grad_points + model.poly_basis_size(), direct_op_weights.rows());
  EXPECT_EQ(n_points + dim * n_grad_points + model.poly_basis_size(), op_weights.rows());

  auto max_residual = (op_weights - direct_op_weights).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

}  // namespace

TEST(rbf_operator, trivial) {
  test_poly_degree(0);
  test_poly_degree(1);
  test_poly_degree(2);
}
