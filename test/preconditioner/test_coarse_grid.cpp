#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <polatory/interpolation/direct_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <random>
#include <utility>
#include <vector>

#include "../utility.hpp"

using polatory::Index;
using polatory::Mat;
using polatory::Model;
using polatory::VecX;
using polatory::interpolation::DirectEvaluator;
using polatory::numeric::relative_error;
using polatory::preconditioner::CoarseGrid;
using polatory::preconditioner::Domain;
using polatory::rbf::Triharmonic3D;

namespace {

template <int Dim>
void test(Index n_points, Index n_grad_points) {
  constexpr int kDim = Dim;
  using Domain = Domain<kDim>;
  using Mat = Mat<kDim>;

  auto relative_tolerance = 1e-8;

  Mat aniso = Mat::Identity();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  Triharmonic3D<kDim> rbf({1.0});

  auto poly_degree = rbf.cpd_order() - 1;
  Model<kDim> model(std::move(rbf), poly_degree);
  model.set_nugget(0.01);

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
  }

  CoarseGrid<kDim> coarse(model, std::move(domain));
  coarse.setup(points, grad_points);

  VecX rhs = VecX(mu + kDim * sigma);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();
  coarse.solve(rhs);

  VecX sol = VecX::Zero(mu + kDim * sigma + l);
  coarse.set_solution_to(sol);

  DirectEvaluator<kDim> eval(model, points, grad_points);
  eval.set_weights(sol);
  eval.set_target_points(points, grad_points);

  VecX values_fit = eval.evaluate();
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
