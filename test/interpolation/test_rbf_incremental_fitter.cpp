// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/biharmonic.hpp>

#include "sample_data.hpp"

using polatory::common::take_rows;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_incremental_fitter;
using polatory::polynomial::basis_base;
using polatory::rbf::biharmonic;

TEST(rbf_incremental_fitter, trivial) {
  const size_t n_surface_points = 4096;
  const int poly_dimension = 3;
  const int poly_degree = 0;
  double absolute_tolerance = 1e-4;

  points3d points;
  valuesd values;
  std::tie(points, values) = sample_sdf_data(n_surface_points);

  biharmonic rbf({ 1.0, 0.0 });

  std::vector<size_t> indices;
  valuesd weights;

  rbf_incremental_fitter fitter(rbf, poly_dimension, poly_degree, points);
  std::tie(indices, weights) = fitter.fit(values, absolute_tolerance);

  size_t n_poly_basis = basis_base::basis_size(poly_dimension, poly_degree);
  EXPECT_EQ(weights.rows(), indices.size() + n_poly_basis);

  rbf_evaluator<> eval(rbf, poly_dimension, poly_degree, take_rows(points, indices));
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate_points(points);

  auto max_residual = (values - values_fit).lpNorm<Eigen::Infinity>();
  std::cout << "Maximum residual:" << std::endl
            << "  " << max_residual << std::endl;

  EXPECT_LT(max_residual, absolute_tolerance);
}
