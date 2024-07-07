#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <utility>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::vectord;
using polatory::interpolation::rbf_fitter;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::numeric::absolute_error;
using polatory::rbf::triharmonic3d;

namespace {

void test(index_t n_points, index_t n_grad_points) {
  constexpr int kDim = 3;
  auto absolute_tolerance = 1e-3;
  auto grad_absolute_tolerance = 1e-2;
  auto max_iter = 100;
  auto accuracy = absolute_tolerance / 100.0;
  auto grad_accuracy = grad_absolute_tolerance / 100.0;

  auto aniso = random_anisotropy<kDim>();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  vectord rhs(n_points + kDim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  triharmonic3d<kDim> rbf({1.0});
  rbf.set_anisotropy(aniso);

  auto poly_degree = rbf.cpd_order() - 1;
  model<kDim> model(std::move(rbf), poly_degree);
  model.set_nugget(0.01);

  rbf_fitter<kDim> fitter(model, points, grad_points);
  vectord weights = fitter.fit(rhs, absolute_tolerance, grad_absolute_tolerance, max_iter, accuracy,
                               grad_accuracy);

  EXPECT_EQ(weights.rows(), n_points + kDim * n_grad_points + model.poly_basis_size());

  rbf_symmetric_evaluator<kDim> eval(model, points, grad_points, accuracy, grad_accuracy);
  eval.set_weights(weights);

  vectord values_fit = eval.evaluate();
  values_fit.head(n_points) += weights.head(n_points) * model.nugget();

  EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.head(n_points), rhs.head(n_points)),
            absolute_tolerance);

  if (n_grad_points > 0) {
    EXPECT_LT(absolute_error<Eigen::Infinity>(values_fit.tail(kDim * n_grad_points),
                                              rhs.tail(kDim * n_grad_points)),
              grad_absolute_tolerance);
  }
}

}  // namespace

TEST(rbf_fitter, trivial) { test(10000, 10000); }

TEST(rbf_fitter, special_case) { test(1, 10000); }
