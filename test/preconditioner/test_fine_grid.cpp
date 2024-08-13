#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/binary_cache.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <random>
#include <utility>
#include <vector>

#include "../utility.hpp"

using polatory::Index;
using polatory::Mat;
using polatory::MatX;
using polatory::Model;
using polatory::VecX;
using polatory::numeric::relative_error;
using polatory::polynomial::LagrangeBasis;
using polatory::preconditioner::BinaryCache;
using polatory::preconditioner::CoarseGrid;
using polatory::preconditioner::Domain;
using polatory::preconditioner::FineGrid;
using polatory::rbf::Triharmonic3D;

namespace {

template <int Dim>
void test(Index n_points, Index n_grad_points) {
  constexpr int kDim = Dim;
  using Domain = Domain<kDim>;
  using LagrangeBasis = LagrangeBasis<kDim>;
  using Mat = Mat<kDim>;

  auto relative_tolerance = 1e-8;

  Mat aniso = Mat::Identity();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  Triharmonic3D<kDim> rbf({1.0});

  auto poly_degree = rbf.cpd_order() - 1;
  Model<kDim> model(std::move(rbf), poly_degree);

  auto mu = n_points;
  auto sigma = n_grad_points;
  auto l = model.poly_basis_size();

  Domain domain;
  {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<Index> indices(mu);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<Index> grad_indices(sigma);
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

  MatX lagrange_p;
  if (l > 0) {
    if (poly_degree == 1 && mu == 1 && sigma >= 1) {
      // The special case.
      LagrangeBasis lagrange_basis(poly_degree, points, grad_points.topRows(1));
      lagrange_p = lagrange_basis.evaluate(points, grad_points);
    } else {
      // The ordinary case.
      std::vector<Index> poly_point_idcs(domain.point_indices.begin(),
                                         domain.point_indices.begin() + l);
      LagrangeBasis lagrange_basis(poly_degree, points(poly_point_idcs, Eigen::all));
      lagrange_p = lagrange_basis.evaluate(points, grad_points);
    }
  }

  CoarseGrid<kDim> coarse(model, std::move(domain_coarse));
  BinaryCache cache;
  FineGrid<kDim> fine(model, std::move(domain_fine), cache);
  coarse.setup(points, grad_points, lagrange_p);
  fine.setup(points, grad_points, lagrange_p);

  VecX rhs = VecX(mu + kDim * sigma);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();
  coarse.solve(rhs);
  fine.solve(rhs);

  VecX sol_coarse = VecX::Zero(mu + kDim * sigma + l);
  coarse.set_solution_to(sol_coarse);
  for (Index i = 0; i < mu; i++) {
    if (!domain.inner_point.at(i)) {
      sol_coarse(domain.point_indices.at(i)) = 0.0;
    }
  }
  for (Index i = 0; i < sigma; i++) {
    if (!domain.inner_grad_point.at(i)) {
      sol_coarse.segment<kDim>(mu + kDim * domain.grad_point_indices.at(i)).array() = 0.0;
    }
  }
  sol_coarse.tail(l).array() = 0.0;

  VecX sol_fine = VecX::Zero(mu + kDim * sigma + l);
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
