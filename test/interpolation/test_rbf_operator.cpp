#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>
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
using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_operator;
using polatory::interpolation::rbf_operator;
using polatory::point_cloud::random_points;
using polatory::rbf::multiquadric1;

namespace {

void test_poly_degree(int poly_degree, index_t n_initial_points, index_t n_initial_grad_points) {
  const int dim = 3;
  index_t n_points = n_initial_points;
  index_t n_grad_points = n_initial_grad_points;

  auto absolute_tolerance = 5e-6;
  auto grad_absolute_tolerance = 5e-5;
  auto poly_absolute_tolerance = 1e-10;

  multiquadric1 rbf({1.0, 0.001});
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, dim, poly_degree);
  model.set_nugget(0.01);

  bbox3d bbox{-point3d::Ones(), point3d::Ones()};
  rbf_operator<> op(model, bbox);

  for (auto i = 0; i < 4; i++) {
    auto points = random_points(sphere3d(), n_points);
    auto grad_points = random_points(sphere3d(), n_grad_points);

    valuesd weights = valuesd::Random(n_points + dim * n_grad_points + model.poly_basis_size());

    rbf_direct_operator direct_op(model, points, grad_points);
    op.set_points(points, grad_points);

    valuesd direct_op_weights = direct_op(weights);
    valuesd op_weights = op(weights);

    EXPECT_EQ(n_points + dim * n_grad_points + model.poly_basis_size(), direct_op_weights.rows());
    EXPECT_EQ(n_points + dim * n_grad_points + model.poly_basis_size(), op_weights.rows());

    auto max_residual = (op_weights - direct_op_weights).head(n_points).lpNorm<Eigen::Infinity>();
    EXPECT_LT(max_residual, absolute_tolerance);

    if (n_grad_points > 0) {
      auto grad_max_residual = (op_weights - direct_op_weights)
                                   .segment(n_points, dim * n_grad_points)
                                   .lpNorm<Eigen::Infinity>();
      EXPECT_LT(grad_max_residual, grad_absolute_tolerance);
    }

    auto poly_max_residual =
        (op_weights - direct_op_weights).tail(model.poly_basis_size()).lpNorm<Eigen::Infinity>();
    EXPECT_LT(poly_max_residual, poly_absolute_tolerance);

    n_points *= 2;
    n_grad_points *= 2;
  }
}

}  // namespace

TEST(rbf_operator, trivial) {
  test_poly_degree(0, 512, 0);
  test_poly_degree(0, 0, 128);

  test_poly_degree(0, 512, 128);
  test_poly_degree(1, 512, 128);
  test_poly_degree(2, 512, 128);
}
