// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "distribution_generator/spherical_distribution.hpp"
#include "interpolation/rbf_fitter.hpp"
#include "interpolation/rbf_symmetric_evaluator.hpp"
#include "polynomial/basis_base.hpp"
#include "rbf/linear_variogram.hpp"

using namespace polatory::interpolation;
using polatory::distribution_generator::spherical_distribution;
using polatory::polynomial::basis_base;
using polatory::rbf::linear_variogram;

namespace {

auto test_points() {
  size_t n_points = 30000;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double radius = 1.0;//1e5;

  auto n_mones = n_points / 4;
  auto n_zeros = n_points / 2;
  auto n_ones = n_points - (n_mones + n_zeros);

  auto pt_mones = spherical_distribution(n_mones, center, 0.9 * radius);
  auto pt_zeros = spherical_distribution(n_zeros, center, 0.95 * radius);
  auto pt_ones = spherical_distribution(n_ones, center, radius);

  std::vector<Eigen::Vector3d> points;
  points.reserve(n_points);

  points.insert(points.end(), pt_mones.begin(), pt_mones.end());
  points.insert(points.end(), pt_zeros.begin(), pt_zeros.end());
  points.insert(points.end(), pt_ones.begin(), pt_ones.end());

  Eigen::VectorXd values = Eigen::VectorXd::Zero(n_points);
  values.head(n_mones) = Eigen::VectorXd::Constant(n_mones, -1.0);
  values.tail(n_ones) = Eigen::VectorXd::Constant(n_ones, 1.0);

  return std::make_pair(std::move(points), std::move(values));
}

void test_poly_degree(int poly_degree, bool with_initial_solution) {
  std::vector<Eigen::Vector3d> points;
  Eigen::VectorXd values;
  std::tie(points, values) = test_points();

  size_t n_points = points.size();
  size_t n_polynomials = basis_base::dimension(poly_degree);
  double absolute_tolerance = 1e-4;

  linear_variogram rbf({ 1.0, 0.0 });

  auto fitter = std::make_unique<rbf_fitter>(rbf, poly_degree, points);
  Eigen::VectorXd weights;
  if (with_initial_solution) {
    Eigen::VectorXd x0 = 1e-5 * Eigen::VectorXd::Random(n_points + n_polynomials);
    weights = fitter->fit(values, absolute_tolerance, x0);
  } else {
    weights = fitter->fit(values, absolute_tolerance);
  }
  fitter.reset();

  rbf_symmetric_evaluator<> eval(rbf, poly_degree, points);
  eval.set_weights(weights);
  Eigen::VectorXd values_fit = eval.evaluate();

  Eigen::VectorXd residuals = (values - values_fit).cwiseAbs();
  Eigen::VectorXd nuggets = rbf.nugget() * weights.head(n_points).cwiseAbs();

  std::cout << "Maximum residual:" << std::endl
            << "  " << residuals.lpNorm<Eigen::Infinity>() << std::endl;

  for (size_t i = 0; i < n_points; i++) {
    EXPECT_LT(residuals(i), absolute_tolerance + nuggets(i));
  }
}

} // namespace

TEST(rbf_fitter, trivial) {
  test_poly_degree(-1, false);
  test_poly_degree(-1, true);
  test_poly_degree(0, false);
  test_poly_degree(0, true);
  test_poly_degree(1, false);
  test_poly_degree(1, true);
  //test_poly_degree(2, false);
  //test_poly_degree(2, true);
}
