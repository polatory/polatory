#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/interpolation/fitter.hpp>
#include <polatory/interpolation/symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <utility>

#include "../utility.hpp"

using polatory::Index;
using polatory::Model;
using polatory::VecX;
using polatory::interpolation::Fitter;
using polatory::interpolation::SymmetricEvaluator;
using polatory::numeric::absolute_error;
using polatory::rbf::Triharmonic3D;

namespace {

void test(Index n_points, Index n_grad_points) {
  constexpr int kDim = 3;
  auto tolerance = 1e-3;
  auto grad_tolerance = 1e-2;
  auto max_iter = 100;
  auto accuracy = tolerance / 100.0;
  auto grad_accuracy = grad_tolerance / 100.0;

  auto aniso = random_anisotropy<kDim>();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  VecX rhs(n_points + kDim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  Triharmonic3D<kDim> rbf({1.0});
  rbf.set_anisotropy(aniso);

  auto poly_degree = rbf.cpd_order() - 1;
  Model<kDim> model(std::move(rbf), poly_degree);
  model.set_nugget(0.01);

  Fitter<kDim> fitter(model, points, grad_points);
  VecX weights = fitter.fit(rhs, tolerance, grad_tolerance, max_iter, accuracy, grad_accuracy);

  EXPECT_EQ(weights.rows(), n_points + kDim * n_grad_points + model.poly_basis_size());

  SymmetricEvaluator<kDim> eval(model, points, grad_points, accuracy, grad_accuracy);
  eval.set_weights(weights);

  VecX values_fit = eval.evaluate();
  values_fit.head(n_points) += weights.head(n_points) * model.nugget();

  EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.head(n_points), rhs.head(n_points)),
            tolerance);

  if (n_grad_points > 0) {
    EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.tail(kDim * n_grad_points),
                                              rhs.tail(kDim * n_grad_points)),
              grad_tolerance);
  }
}

}  // namespace

TEST(rbf_fitter, values) { test(10000, 0); }

TEST(rbf_fitter, values_and_grads) { test(10000, 10000); }

TEST(rbf_fitter, special_case) { test(1, 10000); }
