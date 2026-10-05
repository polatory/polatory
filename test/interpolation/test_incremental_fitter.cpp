#include <gtest/gtest.h>

#include <Eigen/Core>
#include <jizai/interpolation/evaluator.hpp>
#include <jizai/interpolation/incremental_fitter.hpp>
#include <jizai/model.hpp>
#include <jizai/numeric/error.hpp>
#include <jizai/rbf/polyharmonic_odd.hpp>
#include <jizai/types.hpp>
#include <utility>

#include "../utility.hpp"

using jizai::Index;
using jizai::kAll;
using jizai::Model;
using jizai::VecX;
using jizai::interpolation::Evaluator;
using jizai::interpolation::IncrementalFitter;
using jizai::numeric::absolute_error;
using jizai::rbf::Triharmonic3D;

TEST(rbf_incremental_fitter, trivial) {
  constexpr int kDim = 3;

  Index n_points = 10000;
  Index n_grad_points = 10000;
  auto tolerance = 1e-2;
  auto grad_tolerance = 1e-1;
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

  IncrementalFitter<kDim> fitter(model, points, grad_points);
  auto [indices, grad_indices, weights] =
      fitter.fit(rhs, tolerance, grad_tolerance, max_iter, accuracy, grad_accuracy);

  EXPECT_EQ(weights.rows(), indices.size() + kDim * grad_indices.size() + model.poly_basis_size());

  Evaluator<kDim> eval(model, points(indices, kAll), grad_points(grad_indices, kAll), accuracy,
                       grad_accuracy);
  eval.set_weights(weights);
  eval.set_target_points(points, grad_points);
  VecX values_fit = eval.evaluate();

  EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.head(n_points), rhs.head(n_points)),
            tolerance);

  if (n_grad_points > 0) {
    EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.tail(kDim * n_grad_points),
                                              rhs.tail(kDim * n_grad_points)),
              grad_tolerance);
  }
}
