#include <gtest/gtest.h>

#include <Eigen/Core>
#include <format>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/precision.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::matrixNd;
using polatory::interpolation::rbf_direct_evaluator;
using polatory::interpolation::rbf_fitter;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_spheroidal9;
using polatory::rbf::triharmonic3d;

namespace {

template <class Rbf>
void test(Rbf rbf, int poly_degree, index_t n_points, index_t n_grad_points = 0) {
  constexpr int kDim = Rbf::kDim;
  std::cout << std::format("dim: {}, deg: {}, n_points: {}, n_grad_points: {}", kDim, poly_degree,
                           n_points, n_grad_points)
            << std::endl;

  auto absolute_tolerance = 1e-4;
  auto grad_absolute_tolerance = 1e-4;
  auto max_iter = 32;

  matrixNd<kDim> aniso = matrixNd<kDim>::Identity();
  auto [points, values] = sample_data(n_points, aniso);
  auto [grad_points, grad_values] = sample_grad_data(n_grad_points, aniso);

  valuesd rhs(n_points + kDim * n_grad_points);
  rhs << values, grad_values.template reshaped<Eigen::RowMajor>();

  rbf.set_anisotropy(aniso);

  model model(rbf, poly_degree);
  model.set_nugget(0.01);

  rbf_fitter fitter(model, points, grad_points);
  valuesd weights = fitter.fit(rhs, absolute_tolerance, grad_absolute_tolerance, max_iter);

  EXPECT_EQ(weights.rows(), n_points + kDim * n_grad_points + model.poly_basis_size());

  rbf_direct_evaluator eval(model, points, grad_points);
  eval.set_weights(weights);
  eval.set_target_points(points, grad_points);
  valuesd values_fit = eval.evaluate();
  values_fit.head(n_points) += weights.head(n_points) * model.nugget();

  auto max_residual = (rhs - values_fit).head(n_points).template lpNorm<Eigen::Infinity>();
  std::cout << std::format("Absolute residual (exact): {}", max_residual) << std::endl;
  EXPECT_LT(max_residual, absolute_tolerance);

  if (n_grad_points > 0) {
    auto max_grad_residual =
        (rhs - values_fit).tail(kDim * n_grad_points).template lpNorm<Eigen::Infinity>();
    std::cout << std::format("Absolute grad residual (exact): {}", max_grad_residual) << std::endl;
    EXPECT_LT(max_grad_residual, grad_absolute_tolerance);
  }
}

}  // namespace

TEST(rbf_fitter, trivial) {
  test(cov_spheroidal9<1>({1.0, 0.01}), -1, 4096);
  test(cov_spheroidal9<2>({1.0, 0.01}), -1, 4096);
  test(cov_spheroidal9<3>({1.0, 0.01}), -1, 4096);
  test(biharmonic3d<1>({1.0}), 0, 4096);
  test(biharmonic3d<2>({1.0}), 0, 4096);
  test(biharmonic3d<3>({1.0}), 0, 4096);
  test(biharmonic3d<1>({1.0}), 1, 4096);
  test(biharmonic3d<2>({1.0}), 1, 4096);
  test(biharmonic3d<3>({1.0}), 1, 4096);
  test(biharmonic3d<1>({1.0}), 2, 4096);
  test(biharmonic3d<2>({1.0}), 2, 4096);
  test(biharmonic3d<3>({1.0}), 2, 4096);
  test(triharmonic3d<1>({1.0}), 1, 1, 1024);
  test(triharmonic3d<2>({1.0}), 1, 1, 1024);
  test(triharmonic3d<3>({1.0}), 1, 1, 1024);
}
