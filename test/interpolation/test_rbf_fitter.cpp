// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <memory>
#include <tuple>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/biharmonic.hpp>

#include "test_points_values.hpp"

using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_fitter;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::polynomial::basis_base;
using polatory::rbf::biharmonic;

namespace {

void test_poly_degree(int poly_degree, bool with_initial_solution) {
  points3d points;
  valuesd values;
  std::tie(points, values) = test_points_values(30000);

  size_t n_points = points.rows();
  size_t n_poly_basis = basis_base::basis_size(3, poly_degree);
  double absolute_tolerance = 1e-4;

  biharmonic rbf({ 1.0, 0.0 });

  auto fitter = std::make_unique<rbf_fitter>(rbf, 3, poly_degree, points);
  valuesd weights;
  if (with_initial_solution) {
    valuesd x0 = 1e-5 * valuesd::Random(n_points + n_poly_basis);
    weights = fitter->fit(values, absolute_tolerance, x0);
  } else {
    weights = fitter->fit(values, absolute_tolerance);
  }
  fitter.reset();

  rbf_symmetric_evaluator<> eval(rbf, 3, poly_degree, points);
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate();

  valuesd residuals = (values - values_fit).cwiseAbs();
  valuesd smoothing_error_bounds = rbf.nugget() * weights.head(n_points).cwiseAbs();

  std::cout << "Maximum residual:" << std::endl
            << "  " << residuals.lpNorm<Eigen::Infinity>() << std::endl;

  for (size_t i = 0; i < n_points; i++) {
    EXPECT_LT(residuals(i), absolute_tolerance + smoothing_error_bounds(i));
  }
}

}  // namespace

TEST(rbf_fitter, trivial) {
  test_poly_degree(0, false);
  test_poly_degree(0, true);
  test_poly_degree(1, false);
  test_poly_degree(1, true);
  test_poly_degree(2, false);
  test_poly_degree(2, true);
}
