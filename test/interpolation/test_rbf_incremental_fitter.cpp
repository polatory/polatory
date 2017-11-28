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

#include "test_points_values.hpp"

using namespace polatory::interpolation;
using polatory::common::take_rows;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::polynomial::basis_base;
using polatory::rbf::biharmonic;

namespace {

void test_poly_degree(int poly_degree) {
  points3d points;
  valuesd values;
  std::tie(points, values) = test_points_values(10000);

  size_t n_poly_basis = basis_base::basis_size(3, poly_degree);
  double absolute_tolerance = 1e-4;

  biharmonic rbf({ 1.0, 0.0 });

  auto fitter = std::make_unique<rbf_incremental_fitter>(rbf, 3, poly_degree, points);
  std::vector<size_t> point_indices;
  valuesd weights;

  std::tie(point_indices, weights) = fitter->fit(values, absolute_tolerance);
  EXPECT_EQ(weights.rows(), point_indices.size() + n_poly_basis);
  fitter.reset();

  rbf_evaluator<> eval(rbf, 3, poly_degree, take_rows(points, point_indices));
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate_points(points);

  auto max_residual = (values - values_fit).lpNorm<Eigen::Infinity>();
  std::cout << "Maximum residual:" << std::endl
            << "  " << max_residual << std::endl;

  EXPECT_LT(max_residual, absolute_tolerance);
}

} // namespace

TEST(rbf_incremental_fitter, trivial) {
  test_poly_degree(0);
}
