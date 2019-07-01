// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <tuple>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/biharmonic3d.hpp>

#include "random_transformation.hpp"
#include "sample_data.hpp"

using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_fitter;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::model;
using polatory::rbf::biharmonic3d;

namespace {

void test_poly_degree(int poly_degree) {
  const size_t n_surface_points = 10000;
  const int poly_dimension = 3;
  double absolute_tolerance = 1e-4;

  points3d points;
  valuesd values;
  std::tie(points, values) = sample_sdf_data(n_surface_points);

  size_t n_points = points.rows();

  biharmonic3d rbf({ 1.0 });
  rbf.set_transformation(random_transformation());

  model model(rbf, poly_dimension, poly_degree);

  rbf_fitter fitter(model, points);
  valuesd weights = fitter.fit(values, absolute_tolerance);

  EXPECT_EQ(weights.rows(), n_points + model.poly_basis_size());

  rbf_symmetric_evaluator<> eval(model, points);
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate();

  valuesd residuals = (values - values_fit).cwiseAbs();
  valuesd smoothing_error_bounds = model.nugget() * weights.head(n_points).cwiseAbs();

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
