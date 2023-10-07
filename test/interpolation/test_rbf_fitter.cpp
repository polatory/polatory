#include <gtest/gtest.h>

#include <Eigen/Core>
#include <format>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
#include <polatory/rbf/multiquadric.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_fitter;
using polatory::rbf::inverse_multiquadric1;
using polatory::rbf::multiquadric1;
using polatory::rbf::multiquadric3;
using polatory::rbf::multiquadric5;

namespace {

template <class Rbf>
void test(int poly_degree) {
  constexpr int kDim = Rbf::kDim;
  std::cout << std::format("dim: {}, deg: {}", kDim, poly_degree) << std::endl;

  using Model = model<Rbf>;

  index_t n_points = 1000;
  index_t n_grad_points = 100;

  auto absolute_tolerance = 1e-3;
  auto grad_absolute_tolerance = 1e-1;
  auto max_iter = 32;

  auto aniso = random_anisotropy<kDim>();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  valuesd rhs(n_points + kDim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  Rbf rbf({1.0, 1e-4});
  rbf.set_anisotropy(aniso);

  Model model(rbf, poly_degree);
  // model.set_nugget(0.01);

  rbf_fitter fitter(model, points, grad_points);
  valuesd weights = fitter.fit(rhs, absolute_tolerance, grad_absolute_tolerance, max_iter);

  EXPECT_EQ(weights.rows(), n_points + kDim * n_grad_points + model.poly_basis_size());

  rbf_direct_evaluator<Model> eval(model, points, grad_points);
  eval.set_weights(weights);
  eval.set_target_points(points, grad_points);
  valuesd values_fit = eval.evaluate() + weights.head(n_points) * model.nugget();

  auto max_residual = (rhs - values_fit).head(n_points).template lpNorm<Eigen::Infinity>();
  std::cout << std::format("Max residual: {}", max_residual) << std::endl;
  EXPECT_LT(max_residual, absolute_tolerance);

  auto max_grad_residual =
      (rhs - values_fit).tail(kDim * n_grad_points).template lpNorm<Eigen::Infinity>();
  std::cout << std::format("Max grad residual: {}", max_grad_residual) << std::endl;
  EXPECT_LT(max_grad_residual, grad_absolute_tolerance);
}

}  // namespace

TEST(rbf_fitter, trivial) {
  test<inverse_multiquadric1<1>>(-1);
  test<inverse_multiquadric1<2>>(-1);
  test<inverse_multiquadric1<3>>(-1);
  test<multiquadric1<1>>(0);
  test<multiquadric1<2>>(0);
  test<multiquadric1<3>>(0);
  test<multiquadric3<1>>(1);
  test<multiquadric3<2>>(1);
  test<multiquadric3<3>>(1);
  test<multiquadric5<1>>(2);
  test<multiquadric5<2>>(2);
  test<multiquadric5<3>>(2);
}
