// Copyright (c) 2016, GSI and The Polatory Authors.

#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "polatory/interpolation/rbf_direct_symmetric_evaluator.hpp"
#include "polatory/interpolation/rbf_direct_solver.hpp"
#include "polatory/random_points/box_points.hpp"
#include "polatory/rbf/biharmonic.hpp"

using namespace polatory::interpolation;
using polatory::random_points::box_points;
using polatory::rbf::biharmonic;

namespace {

void test_rbf_direct_solver(double nugget, int poly_degree) {
  size_t n_points = 1024;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double radius = 1e5;
  double absolute_tolerance = 1e-10;

  biharmonic rbf({ 1.0, nugget });

  auto points = box_points(n_points, center, radius);
  Eigen::VectorXd values = Eigen::VectorXd::Random(n_points);

  rbf_direct_solver<double> solver(rbf, poly_degree, points);
  auto weights = solver.solve(values);

  rbf_direct_symmetric_evaluator eval(rbf, poly_degree, points);
  eval.set_weights(weights);
  Eigen::VectorXd values_fit = eval.evaluate();

  Eigen::VectorXd residuals = (values - values_fit).cwiseAbs();
  Eigen::VectorXd nuggets = rbf.nugget() * weights.head(n_points).cwiseAbs();

  for (size_t i = 0; i < n_points; i++) {
    EXPECT_LT(residuals(i), absolute_tolerance + nuggets(i));
  }
}

} // namespace

TEST(rbf_direct_solver, trivial) {
  test_rbf_direct_solver(0.0, -1);
  test_rbf_direct_solver(0.2, -1);
  test_rbf_direct_solver(0.0, 0);
  test_rbf_direct_solver(0.2, 0);
  test_rbf_direct_solver(0.0, 1);
  test_rbf_direct_solver(0.2, 1);
  test_rbf_direct_solver(0.0, 2);
  test_rbf_direct_solver(0.2, 2);
}
