#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <format>
#include <numeric>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/rbf/multiquadric1.hpp>
#include <polatory/types.hpp>
#include <random>
#include <utility>
#include <vector>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::matrixNd;
using polatory::geometry::pointsNd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::polynomial::lagrange_basis;
using polatory::preconditioner::coarse_grid;
using polatory::preconditioner::domain;
using polatory::rbf::multiquadric1;

namespace {

template <int Dim>
void test(int poly_degree) {
  std::cout << std::format("dim: {}, deg: {}", Dim, poly_degree) << std::endl;

  using Rbf = multiquadric1<Dim>;
  using Model = model<Rbf>;
  using Points = pointsNd<Dim>;
  using Matrix = matrixNd<Dim>;
  using Domain = domain<Dim>;
  using LagrangeBasis = lagrange_basis<Dim>;

  index_t mu = 1024;
  index_t sigma = 0;
  auto absolute_tolerance = 1e-10;

  Matrix aniso = Matrix::Identity();

  auto [points, values] = sample_data(mu, aniso);
  auto [grad_points, grad_values] = sample_grad_data(sigma, aniso);

  Rbf rbf({1.0, 1e-3});

  Model model(rbf, poly_degree);
  model.set_nugget(0.01);
  auto l = model.poly_basis_size();

  Domain domain;
  {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<index_t> indices(mu);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<index_t> grad_indices(sigma);
    std::iota(grad_indices.begin(), grad_indices.end(), 0);
    std::shuffle(grad_indices.begin(), grad_indices.end(), gen);

    domain.point_indices = std::move(indices);
    domain.grad_point_indices = std::move(grad_indices);
  }

  std::vector<index_t> poly_point_indices(domain.point_indices.begin(),
                                          domain.point_indices.begin() + l);

  coarse_grid<Model> coarse(model, std::move(domain));
  LagrangeBasis lagrange_basis(poly_degree, points(poly_point_indices, Eigen::all));
  auto lagrange_pt = lagrange_basis.evaluate(points, grad_points);
  coarse.setup(points, grad_points, lagrange_pt);

  valuesd rhs = valuesd(mu + Dim * sigma);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();
  coarse.solve(rhs);

  valuesd sol = valuesd::Zero(mu + Dim * sigma + l);
  coarse.set_solution_to(sol);

  auto eval = rbf_direct_evaluator<Model>(model, points, grad_points);
  eval.set_weights(sol);
  eval.set_target_points(points, Points(0, Dim));
  valuesd values_fit = eval.evaluate() + sol.head(mu) * model.nugget();

  auto max_residual = (rhs.head(mu) - values_fit).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

}  // namespace

TEST(coarse_grid, trivial) {
  for (auto deg = 0; deg <= 2; deg++) {
    test<1>(deg);
    test<2>(deg);
    test<3>(deg);
  }
}
