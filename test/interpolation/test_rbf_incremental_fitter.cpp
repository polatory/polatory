#include <gtest/gtest.h>

#include <Eigen/Core>
#include <format>
#include <iostream>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_incremental_fitter;
using polatory::rbf::inverse_multiquadric1;

namespace {

template <int Dim>
void test(int poly_degree) {
  std::cout << std::format("dim: {}, deg: {}", Dim, poly_degree) << std::endl;

  using Rbf = inverse_multiquadric1<Dim>;
  using Model = model<Rbf>;

  index_t n_points = 1024;
  index_t n_grad_points = 256;

  auto absolute_tolerance = 1e-3;
  auto grad_absolute_tolerance = 1e-3;
  auto max_iter = 32;

  auto aniso = random_anisotropy<Dim>();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  valuesd rhs(n_points + Dim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  Rbf rbf({1.0, 1e-4});
  rbf.set_anisotropy(aniso);

  Model model(rbf, poly_degree);

  rbf_incremental_fitter<Model> fitter(model, points, grad_points);
  auto [indices, grad_indices, weights] =
      fitter.fit(rhs, absolute_tolerance, grad_absolute_tolerance, max_iter);

  EXPECT_EQ(weights.rows(), indices.size() + Dim * grad_indices.size() + model.poly_basis_size());

  rbf_direct_evaluator<Model> eval(model, points(indices, Eigen::all),
                                   grad_points(grad_indices, Eigen::all));
  eval.set_weights(weights);
  eval.set_target_points(points, grad_points);
  valuesd values_fit = eval.evaluate();

  auto max_residual = (rhs - values_fit).head(n_points).template lpNorm<Eigen::Infinity>();
  std::cout << std::format("Max residual: {}", max_residual) << std::endl;
  EXPECT_LT(max_residual, absolute_tolerance);

  auto max_grad_residual =
      (rhs - values_fit).tail(Dim * n_grad_points).template lpNorm<Eigen::Infinity>();
  std::cout << std::format("Max grad residual: {}", max_grad_residual) << std::endl;
  EXPECT_LT(max_grad_residual, grad_absolute_tolerance);
}

}  // namespace

TEST(rbf_incremental_fitter, trivial) {
  for (auto deg = -1; deg <= 2; deg++) {
    test<1>(deg);
    test<2>(deg);
    test<3>(deg);
  }
}
