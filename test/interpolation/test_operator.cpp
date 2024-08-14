#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/direct_operator.hpp>
#include <polatory/interpolation/operator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <utility>

#include "../utility.hpp"

using polatory::Index;
using polatory::Model;
using polatory::VecX;
using polatory::geometry::Points;
using polatory::interpolation::DirectOperator;
using polatory::interpolation::Operator;
using polatory::numeric::absolute_error;
using polatory::rbf::Triharmonic3D;

TEST(rbf_operator, trivial) {
  constexpr int kDim = 3;
  using Points = Points<kDim>;

  Index n_points = 1024;
  Index n_grad_points = 1024;
  auto accuracy = 1e-4;
  auto grad_accuracy = 1e-4;

  Triharmonic3D<kDim> rbf({1.0});
  rbf.set_anisotropy(random_anisotropy<kDim>());

  auto poly_degree = rbf.cpd_order() - 1;
  Model<kDim> model(std::move(rbf), poly_degree);
  model.set_nugget(0.01);

  Points points = Points::Random(n_points, kDim);
  Points grad_points = Points::Random(n_grad_points, kDim);

  VecX weights = VecX::Random(n_points + kDim * n_grad_points + model.poly_basis_size());

  Operator<kDim> op(model, points, grad_points, accuracy, grad_accuracy);

  DirectOperator<kDim> direct_op(model, points, grad_points);

  VecX op_weights = op(weights);
  VecX direct_op_weights = direct_op(weights);

  EXPECT_EQ(n_points + kDim * n_grad_points + model.poly_basis_size(), op_weights.rows());
  EXPECT_EQ(n_points + kDim * n_grad_points + model.poly_basis_size(), direct_op_weights.rows());

  EXPECT_LT(
      absolute_error<Eigen::Infinity>(op_weights.head(n_points), direct_op_weights.head(n_points)),
      accuracy);
  EXPECT_LT(
      absolute_error<Eigen::Infinity>(op_weights.segment(n_points, kDim * n_grad_points),
                                      direct_op_weights.segment(n_points, kDim * n_grad_points)),
      grad_accuracy);
}
