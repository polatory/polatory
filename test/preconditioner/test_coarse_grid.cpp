// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_direct_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/rbf/biharmonic3d.hpp>

using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_symmetric_evaluator;
using polatory::model;
using polatory::point_cloud::random_points;
using polatory::polynomial::lagrange_basis;
using polatory::preconditioner::coarse_grid;
using polatory::rbf::biharmonic3d;

void test_coarse_grid(double nugget) {
  size_t n_points = 1024;
  int poly_dimension = 3;
  int poly_degree = 0;
  double absolute_tolerance = 1e-10;

  auto points = random_points(sphere3d(), n_points);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<size_t> point_indices(n_points);
  std::iota(point_indices.begin(), point_indices.end(), 0);
  std::shuffle(point_indices.begin(), point_indices.end(), gen);

  model model(biharmonic3d({ 1.0, nugget }), poly_dimension, poly_degree);
  auto lagr_basis = std::make_shared<lagrange_basis>(poly_dimension, poly_degree, points.topRows(model.poly_basis_size()));

  coarse_grid coarse(model, lagr_basis, point_indices, points);

  valuesd values = valuesd::Random(n_points);
  coarse.solve(values);

  valuesd sol = valuesd::Zero(n_points + model.poly_basis_size());
  coarse.set_solution_to(sol);

  auto eval = rbf_direct_symmetric_evaluator(model, points);
  eval.set_weights(sol);
  valuesd values_fit = eval.evaluate();

  valuesd residuals = (values - values_fit).cwiseAbs();
  valuesd smoothing_error_bounds = model.rbf().nugget() * sol.head(n_points).cwiseAbs();

  for (size_t i = 0; i < n_points; i++) {
    EXPECT_LT(residuals(i), absolute_tolerance + smoothing_error_bounds(i));
  }
}

TEST(coarse_grid, trivial) {
  test_coarse_grid(0.0);
  test_coarse_grid(0.2);
}
