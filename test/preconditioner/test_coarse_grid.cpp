#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <polatory/common/types.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
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
using polatory::interpolation::rbf_direct_evaluator;
using polatory::numeric::relative_error;
using polatory::polynomial::lagrange_basis;
using polatory::preconditioner::coarse_grid;
using polatory::preconditioner::domain;
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
  model.set_nugget(0.01);

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
  }

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

  coarse_grid<kDim> coarse(model, std::move(domain));
  coarse.setup(points, grad_points, lagrange_pt);

  valuesd rhs = valuesd(mu + kDim * sigma);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();
  coarse.solve(rhs);

  valuesd sol = valuesd::Zero(mu + kDim * sigma + l);
  coarse.set_solution_to(sol);

  rbf_direct_evaluator<kDim> eval(model, points, grad_points);
  eval.set_weights(sol);
  eval.set_target_points(points, grad_points);

  valuesd values_fit = eval.evaluate();
  values_fit.head(mu) += sol.head(mu) * model.nugget();

  EXPECT_LT(relative_error(values_fit, rhs), relative_tolerance);
}

}  // namespace

TEST(coarse_grid, trivial) {
  test<1>(1000, 0);
  test<2>(1000, 0);
  test<3>(1000, 0);
}

TEST(coarse_grid, special_case) {
  test<1>(1, 300);
  test<2>(1, 300);
  test<3>(1, 300);
}
