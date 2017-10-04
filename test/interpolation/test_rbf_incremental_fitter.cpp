// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/common/vector_view.hpp"
#include "polatory/interpolation/rbf_evaluator.hpp"
#include "polatory/interpolation/rbf_incremental_fitter.hpp"
#include "polatory/polynomial/basis_base.hpp"
#include "polatory/rbf/biharmonic.hpp"
#include "test_points_values.hpp"

using namespace polatory::interpolation;
using polatory::common::make_view;
using polatory::polynomial::basis_base;
using polatory::rbf::biharmonic;

namespace {

void test_poly_degree(int poly_degree) {
  std::vector<Eigen::Vector3d> points;
  Eigen::VectorXd values;
  std::tie(points, values) = test_points_values(10000);

  size_t n_polynomials = basis_base::dimension(poly_degree);
  double absolute_tolerance = 1e-4;

  biharmonic rbf({ 1.0, 0.0 });

  auto fitter = std::make_unique<rbf_incremental_fitter>(rbf, poly_degree, points);
  std::vector<size_t> point_indices;
  Eigen::VectorXd weights;

  std::tie(point_indices, weights) = fitter->fit(values, absolute_tolerance);
  EXPECT_EQ(weights.size(), point_indices.size() + n_polynomials);
  fitter.reset();

  rbf_evaluator<> eval(rbf, poly_degree, make_view(points, point_indices));
  eval.set_weights(weights);
  Eigen::VectorXd values_fit = eval.evaluate_points(points);

  auto max_residual = (values - values_fit).lpNorm<Eigen::Infinity>();
  std::cout << "Maximum residual:" << std::endl
            << "  " << max_residual << std::endl;

  EXPECT_LT(max_residual, absolute_tolerance);
}

} // namespace

TEST(rbf_incremental_fitter, trivial) {
  test_poly_degree(0);
}
