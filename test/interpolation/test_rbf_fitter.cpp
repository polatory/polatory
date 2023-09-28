#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/rbf/multiquadric1.hpp>
#include <polatory/rbf/reference/triharmonic3d.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"
#include "utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::geometry::sphere3d;
using polatory::interpolation::rbf_fitter;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::random_points;
using polatory::rbf::multiquadric1;
using polatory::rbf::reference::triharmonic3d;

namespace {

void test_poly_degree(int poly_degree) {
  const int dim = 3;
  const index_t n_surface_points = 10000;
  index_t n_grad_points = 100;

  auto absolute_tolerance = 1e-4;
  auto grad_absolute_tolerance = 1e-2;
  auto max_iter = 32;

  auto [points, values] = sample_sdf_data(n_surface_points);
  auto n_points = points.rows();

  auto grad_points = random_points(sphere3d(), n_grad_points);
  grad_points = distance_filter(grad_points, 1e-4)(grad_points);
  n_grad_points = grad_points.rows();

  valuesd rhs = valuesd(n_points + dim * n_grad_points);
  rhs << values, grad_points.reshaped<Eigen::RowMajor>();

  multiquadric1 rbf({1.0, 1e-3});
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, dim, poly_degree);
  // model.set_nugget(0.01);

  rbf_fitter fitter(model, points, grad_points);
  valuesd weights = fitter.fit(rhs, absolute_tolerance, grad_absolute_tolerance, max_iter);

  EXPECT_EQ(weights.rows(), n_points + dim * n_grad_points + model.poly_basis_size());

  rbf_symmetric_evaluator<> eval(model, points, grad_points);
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate();  //+ weights.head(n_points) * model.nugget();

  auto max_residual = (rhs - values_fit).head(n_points).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);

  auto max_grad_residual = (rhs - values_fit).tail(dim * n_grad_points).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_grad_residual, grad_absolute_tolerance);
}

}  // namespace

TEST(rbf_fitter, trivial) {
  test_poly_degree(0);
  test_poly_degree(1);
  test_poly_degree(2);
}
