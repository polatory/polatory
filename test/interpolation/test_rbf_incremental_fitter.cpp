#include <gtest/gtest.h>

#include <Eigen/Core>
#include <iostream>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"
#include "sample_data.hpp"

using polatory::index_t;
using polatory::model;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_incremental_fitter;
using polatory::rbf::biharmonic3d;

TEST(rbf_incremental_fitter, trivial) {
  const auto n_surface_points = index_t{4096};
  const auto poly_dimension = 3;
  const auto poly_degree = 0;
  const auto absolute_tolerance = 1e-4;

  auto [points, values] = sample_sdf_data(n_surface_points);

  biharmonic3d rbf({1.0});
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, poly_dimension, poly_degree);

  rbf_incremental_fitter fitter(model, points);
  auto [indices, weights] = fitter.fit(values, absolute_tolerance);

  EXPECT_EQ(weights.rows(), indices.size() + model.poly_basis_size());

  rbf_evaluator<> eval(model, points(indices, Eigen::all));
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate(points);

  auto max_residual = (values - values_fit).lpNorm<Eigen::Infinity>();
  std::cout << "Maximum residual:" << std::endl << "  " << max_residual << std::endl;

  EXPECT_LT(max_residual, absolute_tolerance);
}
