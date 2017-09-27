// Copyright (c) 2016, GSI and The Polatory Authors.

#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "distribution_generator/spherical_distribution.hpp"
#include "interpolation/rbf_direct_symmetric_evaluator.hpp"
#include "preconditioner/coarse_grid.hpp"
#include "polynomial/basis_base.hpp"
#include "rbf/linear_variogram.hpp"

using namespace polatory::preconditioner;
using polatory::distribution_generator::spherical_distribution;
using polatory::interpolation::rbf_direct_symmetric_evaluator;
using polatory::polynomial::basis_base;
using polatory::rbf::linear_variogram;

void test_coarse_grid(double nugget) {
  size_t n_points = 1024;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double radius = 1e5;
  int poly_degree = 1;
  size_t n_polynomials = basis_base::dimension(poly_degree);
  double absolute_tolerance = 1e-10;

  auto points = spherical_distribution(n_points, center, radius);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<size_t> point_indices(n_points);
  std::iota(point_indices.begin(), point_indices.end(), 0);
  std::shuffle(point_indices.begin(), point_indices.end(), gen);

  linear_variogram rbf({ 1.0, nugget });

  coarse_grid<double> coarse(rbf, poly_degree, points, point_indices);

  Eigen::VectorXd values = Eigen::VectorXd::Random(n_points);
  coarse.solve(values);

  Eigen::VectorXd sol = Eigen::VectorXd::Zero(n_points + n_polynomials);
  coarse.set_solution_to(sol);

  auto eval = rbf_direct_symmetric_evaluator(rbf, poly_degree, points);
  eval.set_weights(sol);
  Eigen::VectorXd values_fit = eval.evaluate();

  Eigen::VectorXd residuals = (values - values_fit).cwiseAbs();
  Eigen::VectorXd nuggets = rbf.nugget() * sol.head(n_points).cwiseAbs();

  for (size_t i = 0; i < n_points; i++) {
    EXPECT_LT(residuals(i), absolute_tolerance + nuggets(i));
  }
}

TEST(coarse_grid, trivial) {
  test_coarse_grid(0.0);
  test_coarse_grid(0.2);
}
