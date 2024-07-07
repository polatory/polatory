#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <utility>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::vectord;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_incremental_fitter;
using polatory::numeric::absolute_error;
using polatory::rbf::triharmonic3d;

TEST(rbf_incremental_fitter, trivial) {
  constexpr int kDim = 3;

  index_t n_points = 10000;
  index_t n_grad_points = 10000;
  auto tolerance = 1e-2;
  auto grad_tolerance = 1e-1;
  auto max_iter = 100;
  auto accuracy = tolerance / 100.0;
  auto grad_accuracy = grad_tolerance / 100.0;

  auto aniso = random_anisotropy<kDim>();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  vectord rhs(n_points + kDim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  triharmonic3d<kDim> rbf({1.0});
  rbf.set_anisotropy(aniso);

  auto poly_degree = rbf.cpd_order() - 1;
  model<kDim> model(std::move(rbf), poly_degree);

  rbf_incremental_fitter<kDim> fitter(model, points, grad_points);
  auto [indices, grad_indices, weights] =
      fitter.fit(rhs, tolerance, grad_tolerance, max_iter, accuracy, grad_accuracy);

  EXPECT_EQ(weights.rows(), indices.size() + kDim * grad_indices.size() + model.poly_basis_size());

  rbf_evaluator<kDim> eval(model, points(indices, Eigen::all),
                           grad_points(grad_indices, Eigen::all), accuracy, grad_accuracy);
  eval.set_weights(weights);
  eval.set_target_points(points, grad_points);
  vectord values_fit = eval.evaluate();

  EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.head(n_points), rhs.head(n_points)),
            tolerance);

  if (n_grad_points > 0) {
    EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.tail(kDim * n_grad_points),
                                              rhs.tail(kDim * n_grad_points)),
              grad_tolerance);
  }
}
