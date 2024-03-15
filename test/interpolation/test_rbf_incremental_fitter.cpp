#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/precision.hpp>
#include <polatory/rbf/make_rbf.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_incremental_fitter;
using polatory::numeric::absolute_error;
using polatory::rbf::make_rbf;
using polatory::rbf::triharmonic3d;

TEST(rbf_incremental_fitter, trivial) {
  constexpr int kDim = 3;

  index_t n_points = 10000;
  index_t n_grad_points = 10000;
  auto absolute_tolerance = 1e-2;
  auto grad_absolute_tolerance = 1e-1;
  auto max_iter = 100;

  auto aniso = random_anisotropy<kDim>();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  valuesd rhs(n_points + kDim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  auto rbf = make_rbf<triharmonic3d<kDim>>({1.0});
  rbf->set_anisotropy(aniso);

  auto poly_degree = rbf->cpd_order() - 1;
  model<kDim> model(rbf, poly_degree);

  rbf_incremental_fitter<kDim> fitter(model, points, grad_points);
  auto [indices, grad_indices, weights] =
      fitter.fit(rhs, absolute_tolerance, grad_absolute_tolerance, max_iter);

  EXPECT_EQ(weights.rows(), indices.size() + kDim * grad_indices.size() + model.poly_basis_size());

  rbf_evaluator<kDim> eval(model, points(indices, Eigen::all),
                           grad_points(grad_indices, Eigen::all), precision::kPrecise);
  eval.set_weights(weights);
  eval.set_target_points(points, grad_points);
  valuesd values_fit = eval.evaluate();

  EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.head(n_points), rhs.head(n_points)),
            absolute_tolerance);

  if (n_grad_points > 0) {
    EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.tail(kDim * n_grad_points),
                                              rhs.tail(kDim * n_grad_points)),
              grad_absolute_tolerance);
  }
}
