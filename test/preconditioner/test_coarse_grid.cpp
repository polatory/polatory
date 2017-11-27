// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/interpolation/rbf_direct_symmetric_evaluator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/rbf/biharmonic.hpp>

using namespace polatory::preconditioner;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_symmetric_evaluator;
using polatory::point_cloud::random_points;
using polatory::polynomial::basis_base;
using polatory::polynomial::lagrange_basis;
using polatory::rbf::biharmonic;

void test_coarse_grid(double nugget) {
  size_t n_points = 1024;
  int poly_dimension = 3;
  int poly_degree = 0;
  size_t n_poly_basis = basis_base::basis_size(poly_dimension, poly_degree);
  double absolute_tolerance = 1e-10;

  auto points = random_points(sphere3d(), n_points);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<size_t> point_indices(n_points);
  std::iota(point_indices.begin(), point_indices.end(), 0);
  std::shuffle(point_indices.begin(), point_indices.end(), gen);

  auto lagr_basis = std::make_shared<lagrange_basis<double>>(poly_dimension, poly_degree, points.topRows(n_poly_basis));

  biharmonic rbf({ 1.0, nugget });

  coarse_grid<double> coarse(rbf, lagr_basis, point_indices, points);

  Eigen::VectorXd values = Eigen::VectorXd::Random(n_points);
  coarse.solve(values);

  Eigen::VectorXd sol = Eigen::VectorXd::Zero(n_points + n_poly_basis);
  coarse.set_solution_to(sol);

  auto eval = rbf_direct_symmetric_evaluator(rbf, poly_dimension, poly_degree, points);
  eval.set_weights(sol);
  Eigen::VectorXd values_fit = eval.evaluate();

  Eigen::VectorXd residuals = (values - values_fit).cwiseAbs();
  Eigen::VectorXd smoothing_error_bounds = rbf.nugget() * sol.head(n_points).cwiseAbs();

  for (size_t i = 0; i < n_points; i++) {
    EXPECT_LT(residuals(i), absolute_tolerance + smoothing_error_bounds(i));
  }
}

TEST(coarse_grid, trivial) {
  test_coarse_grid(0.0);
  test_coarse_grid(0.2);
}
