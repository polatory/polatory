#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <polatory/common/types.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/binary_cache.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/rbf/make_rbf.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <random>
#include <utility>
#include <vector>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::matrixNd;
using polatory::numeric::relative_error;
using polatory::polynomial::lagrange_basis;
using polatory::preconditioner::binary_cache;
using polatory::preconditioner::coarse_grid;
using polatory::preconditioner::domain;
using polatory::preconditioner::fine_grid;
using polatory::rbf::make_rbf;
using polatory::rbf::triharmonic3d;

namespace {

template <int Dim>
void test(index_t n_points, index_t n_grad_points) {
  constexpr int kDim = Dim;
  using Matrix = matrixNd<kDim>;
  using Domain = domain<kDim>;
  using LagrangeBasis = lagrange_basis<kDim>;

  auto relative_tolerance = 1e-8;

  Matrix aniso = Matrix::Identity();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  auto rbf = make_rbf<triharmonic3d<kDim>>({1.0});

  auto poly_degree = rbf->cpd_order() - 1;
  model<kDim> model(rbf, poly_degree);

  auto mu = n_points;
  auto sigma = n_grad_points;
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

  Eigen::MatrixXd lagrange_pt;
  if (l > 0) {
    if (poly_degree == 1 && mu == 1 && sigma >= 1) {
      // The special case.
      LagrangeBasis lagrange_basis(poly_degree, points, grad_points.topRows(1));
      lagrange_pt = lagrange_basis.evaluate(points, grad_points);
    } else {
      // The ordinary case.
      std::vector<index_t> poly_point_idcs(domain.point_indices.begin(),
                                           domain.point_indices.begin() + l);
      LagrangeBasis lagrange_basis(poly_degree, points(poly_point_idcs, Eigen::all));
      lagrange_pt = lagrange_basis.evaluate(points, grad_points);
    }
  }

  coarse_grid<kDim> coarse(model, std::move(domain_coarse));
  binary_cache cache;
  fine_grid<kDim> fine(model, std::move(domain_fine), cache);
  coarse.setup(points, grad_points, lagrange_pt);
  fine.setup(points, grad_points, lagrange_pt);

  valuesd rhs = valuesd(mu + kDim * sigma);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();
  coarse.solve(rhs);
  fine.solve(rhs);

  valuesd sol_coarse = valuesd::Zero(mu + kDim * sigma + l);
  coarse.set_solution_to(sol_coarse);
  for (index_t i = 0; i < mu; i++) {
    if (!domain.inner_point.at(i)) {
      sol_coarse(domain.point_indices.at(i)) = 0.0;
    }
  }
  for (index_t i = 0; i < sigma; i++) {
    if (!domain.inner_grad_point.at(i)) {
      sol_coarse.segment<kDim>(mu + kDim * domain.grad_point_indices.at(i)).array() = 0.0;
    }
  }
  sol_coarse.tail(l).array() = 0.0;

  valuesd sol_fine = valuesd::Zero(mu + kDim * sigma + l);
  fine.set_solution_to(sol_fine);

  EXPECT_LT(relative_error(sol_fine, sol_coarse), relative_tolerance);
}

}  // namespace

TEST(fine_grid, trivial) {
  test<1>(1000, 0);
  test<2>(1000, 0);
  test<3>(1000, 0);
}

TEST(fine_grid, special_case) {
  test<1>(1, 300);
  test<2>(1, 300);
  test<3>(1, 300);
}
