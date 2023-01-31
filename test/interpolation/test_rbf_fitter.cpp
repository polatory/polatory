#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/types.hpp>
#include <tuple>

#include "../random_anisotropy.hpp"
#include "sample_data.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_fitter;
using polatory::interpolation::rbf_symmetric_evaluator;
using polatory::rbf::biharmonic3d;

namespace {

void test_poly_degree(int poly_degree) {
  const auto n_surface_points = index_t{10000};
  const auto poly_dimension = 3;
  auto absolute_tolerance = 1e-4;

  points3d points;
  valuesd values;
  std::tie(points, values) = sample_sdf_data(n_surface_points);

  auto n_points = points.rows();

  biharmonic3d rbf({1.0});
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, poly_dimension, poly_degree);
  model.set_nugget(0.01);

  rbf_fitter fitter(model, points);
  valuesd weights = fitter.fit(values, absolute_tolerance);

  EXPECT_EQ(weights.rows(), n_points + model.poly_basis_size());

  rbf_symmetric_evaluator<> eval(model, points);
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate() + weights.head(n_points) * model.nugget();

  auto max_residual = (values - values_fit).lpNorm<Eigen::Infinity>();
  EXPECT_LT(max_residual, absolute_tolerance);
}

}  // namespace

TEST(rbf_fitter, trivial) {
  test_poly_degree(0);
  test_poly_degree(1);
  test_poly_degree(2);
}
