#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <format>
#include <numeric>
#include <polatory/model.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
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
using polatory::polynomial::lagrange_basis;
using polatory::preconditioner::coarse_grid;
using polatory::preconditioner::domain;
using polatory::preconditioner::fine_grid;
using polatory::rbf::inverse_multiquadric1;

namespace {

template <int Dim>
void test(int poly_degree) {
  std::cout << std::format("dim: {}, deg: {}", Dim, poly_degree) << std::endl;

  using Rbf = inverse_multiquadric1<Dim>;
  using Matrix = matrixNd<Dim>;
  using Domain = domain<Dim>;
  using LagrangeBasis = lagrange_basis<Dim>;

  index_t mu = 1000;
  index_t sigma = 10;

  Matrix aniso = Matrix::Identity();

  auto [points, values] = sample_data(mu, aniso);
  auto [grad_points, grad_values] = sample_grad_data(sigma, aniso);

  Rbf rbf({1.0, 0.01});

  model model(rbf, poly_degree);
  // model.set_nugget(0.01);
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

    domain.inner_point = std::vector<bool>(mu, true);
    domain.inner_grad_point = std::vector<bool>(sigma, true);
    std::fill(domain.inner_point.begin() + mu / 2, domain.inner_point.end(), false);
    std::fill(domain.inner_grad_point.begin() + sigma / 2, domain.inner_grad_point.end(), false);
  }
  Domain domain_coarse(domain);
  Domain domain_fine(domain);

  std::vector<index_t> poly_point_indices(domain.point_indices.begin(),
                                          domain.point_indices.begin() + l);

  coarse_grid coarse(model, std::move(domain_coarse));
  fine_grid fine(model, std::move(domain_fine));
  Eigen::MatrixXd lagrange_pt;
  if (poly_degree >= 0) {
    LagrangeBasis lagrange_basis(poly_degree, points(poly_point_indices, Eigen::all));
    lagrange_pt = lagrange_basis.evaluate(points, grad_points);
  }
  coarse.setup(points, grad_points, lagrange_pt);
  fine.setup(points, grad_points, lagrange_pt);

  valuesd rhs = valuesd(mu + Dim * sigma);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();
  coarse.solve(rhs);
  fine.solve(rhs);

  valuesd sol_coarse = valuesd::Zero(mu + Dim * sigma + l);
  coarse.set_solution_to(sol_coarse);
  for (index_t i = 0; i < mu; i++) {
    if (!domain.inner_point.at(i)) {
      sol_coarse(domain.point_indices.at(i)) = 0.0;
    }
  }
  for (index_t i = 0; i < sigma; i++) {
    if (!domain.inner_grad_point.at(i)) {
      sol_coarse.segment<Dim>(mu + Dim * domain.grad_point_indices.at(i)).array() = 0.0;
    }
  }
  sol_coarse.tail(l).array() = 0.0;

  valuesd sol_fine = valuesd::Zero(mu + Dim * sigma + l);
  fine.set_solution_to(sol_fine);

  auto max_residual = (sol_coarse - sol_fine).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, 1e-10);
}

}  // namespace

TEST(fine_grid, trivial) {
  for (auto deg = -1; deg <= 2; deg++) {
    test<1>(deg);
    test<2>(deg);
    test<3>(deg);
  }
}
