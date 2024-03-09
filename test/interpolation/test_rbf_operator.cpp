#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/interpolation/rbf_direct_operator.hpp>
#include <polatory/interpolation/rbf_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/precision.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::bbox3d;
using polatory::geometry::point3d;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_operator;
using polatory::interpolation::rbf_operator;
using polatory::point_cloud::random_points;
using polatory::rbf::inverse_multiquadric1;

namespace {

void test(int poly_degree, index_t n_initial_points, index_t n_initial_grad_points) {
  constexpr int kDim = 3;
  using Rbf = inverse_multiquadric1<kDim>;

  index_t n_points = n_initial_points;
  index_t n_grad_points = n_initial_grad_points;

  auto rel_tolerance = 1e-10;

  Rbf rbf({1.0, 0.01});
  rbf.set_anisotropy(random_anisotropy<kDim>());

  model model(rbf, poly_degree);
  model.set_nugget(0.01);

  bbox3d bbox{-point3d::Ones(), point3d::Ones()};
  rbf_operator op(model, bbox, precision::kPrecise);

  for (auto i = 0; i < 4; i++) {
    auto points = random_points(sphere3d(), n_points);
    auto grad_points = random_points(sphere3d(), n_grad_points);

    valuesd weights = valuesd::Random(n_points + kDim * n_grad_points + model.poly_basis_size());

    rbf_direct_operator direct_op(model, points, grad_points);
    op.set_points(points, grad_points);

    valuesd direct_op_weights = direct_op(weights);
    valuesd op_weights = op(weights);

    EXPECT_EQ(n_points + kDim * n_grad_points + model.poly_basis_size(), direct_op_weights.rows());
    EXPECT_EQ(n_points + kDim * n_grad_points + model.poly_basis_size(), op_weights.rows());

    EXPECT_LT(relative_error(op_weights.head(n_points + kDim * n_grad_points),
                             direct_op_weights.head(n_points + kDim * n_grad_points)),
              rel_tolerance);

    EXPECT_EQ(direct_op_weights.tail(model.poly_basis_size()),
              op_weights.tail(model.poly_basis_size()));

    n_points *= 2;
    n_grad_points *= 2;
  }
}

}  // namespace

TEST(rbf_operator, trivial) {
  test(-1, 1024, 0);
  test(-1, 0, 256);

  for (auto deg = -1; deg <= 2; deg++) {
    test(deg, 1024, 256);
  }
}
