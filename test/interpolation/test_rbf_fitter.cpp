// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <tuple>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/biharmonic.hpp>

#include "sample_data.hpp"

using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_fitter;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::polynomial::basis_base;
using polatory::rbf::biharmonic;

namespace {

void test_poly_degree(int poly_degree) {
  const size_t n_surface_points = 10000;
  const int poly_dimension = 3;
  double absolute_tolerance = 1e-4;

  points3d points;
  valuesd values;
  std::tie(points, values) = sample_sdf_data(n_surface_points);

  size_t n_points = points.rows();

  biharmonic rbf({ 1.0, 0.0 });

  rbf_fitter fitter(rbf, poly_dimension, poly_degree, points);
  valuesd weights = fitter.fit(values, absolute_tolerance);

  size_t n_poly_basis = basis_base::basis_size(poly_dimension, poly_degree);
  EXPECT_EQ(weights.rows(), n_points + n_poly_basis);

  rbf_symmetric_evaluator<> eval(rbf, poly_dimension, poly_degree, points);
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
  test_poly_degree(0);
  test_poly_degree(1);
  test_poly_degree(2);
}
