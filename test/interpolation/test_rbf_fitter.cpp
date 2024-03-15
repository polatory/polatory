#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/precision.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::interpolation::rbf_fitter;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::numeric::absolute_error;
using polatory::rbf::RbfPtr;
using polatory::rbf::triharmonic3d;

namespace {

void test(index_t n_points, index_t n_grad_points) {
  constexpr int kDim = 3;
  auto absolute_tolerance = 1e-3;
  auto grad_absolute_tolerance = 1e-2;
  auto max_iter = 100;

  auto aniso = random_anisotropy<kDim>();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  valuesd rhs(n_points + kDim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  RbfPtr<kDim> rbf = std::make_unique<triharmonic3d<kDim>>(std::vector<double>({1.0}));
  rbf->set_anisotropy(aniso);

  auto poly_degree = rbf->cpd_order() - 1;
  model<kDim> model(rbf, poly_degree);
  model.set_nugget(0.01);

  rbf_fitter<kDim> fitter(model, points, grad_points);
  valuesd weights = fitter.fit(rhs, absolute_tolerance, grad_absolute_tolerance, max_iter);

  EXPECT_EQ(weights.rows(), n_points + kDim * n_grad_points + model.poly_basis_size());

  rbf_symmetric_evaluator<kDim> eval(model, points, grad_points, precision::kPrecise);
  eval.set_weights(weights);

  valuesd values_fit = eval.evaluate();
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
