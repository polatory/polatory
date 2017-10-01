// Copyright (c) 2016, GSI and The Polatory Authors.

#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/interpolation/rbf_direct_symmetric_evaluator.hpp"
#include "polatory/preconditioner/fine_grid.hpp"
#include "polatory/rbf/linear_variogram.hpp"

using namespace polatory::preconditioner;
using polatory::interpolation::rbf_direct_symmetric_evaluator;
using polatory::rbf::linear_variogram;

void test_fine_grid(double nugget) {
  size_t n_points = 256;
  double absolute_tolerance = 1e-10;

  std::vector<Eigen::Vector3d> points;
  points.reserve(n_points);

  for (size_t i = 0; i < n_points; i++) {
    points.push_back(Eigen::Vector3d::Random());
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<size_t> point_indices(n_points);
  std::iota(point_indices.begin(), point_indices.end(), 0);
  std::shuffle(point_indices.begin(), point_indices.end(), gen);

  std::vector<bool> inner_point(n_points, true);

  linear_variogram rbf({ 1.0, nugget });

  Eigen::VectorXd values = Eigen::VectorXd::Random(n_points);

  fine_grid<double> fine(rbf, points, point_indices, inner_point);
  fine.solve(values);

  Eigen::VectorXd sol = Eigen::VectorXd::Zero(n_points);
  fine.set_solution_to(sol);

  auto eval = rbf_direct_symmetric_evaluator(rbf, -1, points);
  eval.set_weights(sol);
  Eigen::VectorXd values_fit = eval.evaluate();

  Eigen::VectorXd residuals = (values - values_fit).cwiseAbs();
  Eigen::VectorXd nuggets = rbf.nugget() * sol.head(n_points).cwiseAbs();

  for (size_t i = 0; i < n_points; i++) {
    EXPECT_LT(residuals(i), absolute_tolerance + nuggets(i));
  }
}

TEST(fine_grid, trivial) {
  test_fine_grid(0.0);
  test_fine_grid(0.2);
}
