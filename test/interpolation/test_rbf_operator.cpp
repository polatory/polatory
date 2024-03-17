#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_operator.hpp>
#include <polatory/interpolation/rbf_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/precision.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <utility>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::pointsNd;
using polatory::interpolation::rbf_direct_operator;
using polatory::interpolation::rbf_operator;
using polatory::numeric::relative_error;
using polatory::rbf::triharmonic3d;

TEST(rbf_operator, trivial) {
  constexpr int kDim = 3;
  using Points = pointsNd<kDim>;

  index_t n_points = 1024;
  index_t n_grad_points = 1024;
  auto relative_tolerance = 5e-7;

  triharmonic3d<kDim> rbf({1.0});
  rbf.set_anisotropy(random_anisotropy<kDim>());

  auto poly_degree = rbf.cpd_order() - 1;
  model<kDim> model(std::move(rbf), poly_degree);
  model.set_nugget(0.01);

  Points points = Points::Random(n_points, kDim);
  Points grad_points = Points::Random(n_grad_points, kDim);

  valuesd weights = valuesd::Random(n_points + kDim * n_grad_points + model.poly_basis_size());

  rbf_operator<kDim> op(model, points, grad_points, precision::kPrecise);

  rbf_direct_operator<kDim> direct_op(model, points, grad_points);

  valuesd op_weights = op(weights);
  valuesd direct_op_weights = direct_op(weights);

  EXPECT_EQ(n_points + kDim * n_grad_points + model.poly_basis_size(), op_weights.rows());
  EXPECT_EQ(n_points + kDim * n_grad_points + model.poly_basis_size(), direct_op_weights.rows());

  EXPECT_LT(relative_error(op_weights, direct_op_weights), relative_tolerance);
}
