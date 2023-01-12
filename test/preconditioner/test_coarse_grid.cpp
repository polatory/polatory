#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/types.hpp>

using polatory::common::valuesd;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::model;
using polatory::point_cloud::random_points;
using polatory::polynomial::lagrange_basis;
using polatory::preconditioner::coarse_grid;
using polatory::rbf::biharmonic3d;
using polatory::index_t;

TEST(coarse_grid, trivial) {
  auto n_points = index_t{ 1024 };
  auto poly_dimension = 3;
  auto poly_degree = 0;
  auto absolute_tolerance = 1e-10;

  auto points = random_points(sphere3d(), n_points);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<index_t> point_indices(n_points);
  std::iota(point_indices.begin(), point_indices.end(), 0);
  std::shuffle(point_indices.begin(), point_indices.end(), gen);

  model model(biharmonic3d({ 1.0 }), poly_dimension, poly_degree);
  model.set_nugget(0.01);
  auto lagr_basis = std::make_unique<lagrange_basis>(poly_dimension, poly_degree, points.topRows(model.poly_basis_size()));

  coarse_grid coarse(model, lagr_basis, point_indices, points);

  valuesd values = valuesd::Random(n_points);
  coarse.solve(values);

  valuesd sol = valuesd::Zero(n_points + model.poly_basis_size());
  coarse.set_solution_to(sol);

  auto eval = rbf_direct_evaluator(model, points);
  eval.set_weights(sol);
  eval.set_field_points(points);
  valuesd values_fit = eval.evaluate() + sol.head(n_points) * model.nugget();

  auto max_residual = (values - values_fit).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}
