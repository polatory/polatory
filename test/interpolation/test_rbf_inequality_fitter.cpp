// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_inequality_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/types.hpp>

#include "../random_anisotropy.hpp"
#include "sample_data.hpp"

using polatory::common::take_rows;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_inequality_fitter;
using polatory::model;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;
using polatory::index_t;

TEST(rbf_inequality_fitter, inequality_only) {
  const auto n_points = index_t{ 4096 };
  const auto poly_dimension = 3;
  const auto poly_degree = 0;
  const auto absolute_tolerance = 1e-4;

  points3d points;
  valuesd values;
  std::tie(points, values) = sample_numerical_data(n_points);

  valuesd values_lb = values.array() - 0.5;
  valuesd values_ub = values.array() + 0.5;
  values = valuesd::Constant(n_points, std::numeric_limits<double>::quiet_NaN());

  biharmonic3d rbf({ 1.0 });
  rbf.set_anisotropy(random_anisotropy());

  model model(rbf, poly_dimension, poly_degree);

  std::vector<index_t> indices;
  valuesd weights;

  rbf_inequality_fitter fitter(model, points);
  std::tie(indices, weights) = fitter.fit(values, values_lb, values_ub, absolute_tolerance);

  EXPECT_EQ(weights.rows(), indices.size() + model.poly_basis_size());

  rbf_evaluator<> eval(model, take_rows(points, indices));
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate(points);

  for (index_t i = 0; i < n_points; i++) {
    EXPECT_GT(values_fit(i), values_lb(i) - absolute_tolerance);
    EXPECT_LT(values_fit(i), values_ub(i) + absolute_tolerance);
  }
}

// Example problem taken from https://doi.org/10.1007/BF00897655
TEST(rbf_inequality_fitter, kostov86) {
  const auto n_points = index_t{ 25 };
  const auto poly_dimension = 1;
  const auto poly_degree = -1;
  const auto absolute_tolerance = 1e-5;

  points3d points = points3d::Zero(n_points, 3);
  for (index_t i = 0; i < n_points; i++) {
    points(i, 0) = static_cast<double>(i);
  }

  auto nan = std::numeric_limits<double>::quiet_NaN();
  valuesd values(n_points);
  values <<
    1, nan, 4, nan, 3,
    nan, nan, nan, nan, nan,
    3, nan, nan, 7, nan,
    8, nan, nan, nan, 8,
    3, nan, nan, 6, nan;

  valuesd values_lb(n_points);
  values_lb <<
    nan, 6, nan, 2, nan,
    2, 9, 4, 3, 3,
    nan, nan, 7, nan, 4,
    nan, nan, nan, 5, nan,
    nan, 9, 5, nan, nan;

  valuesd values_ub(n_points);
  values_ub <<
    nan, nan, nan, 4, nan,
    4, nan, nan, nan, nan,
    nan, 1, nan, nan, nan,
    nan, 4, 7, nan, nan,
    nan, nan, nan, nan, 3;

  model model(cov_exponential({ 1.0, 3.0 }), poly_dimension, poly_degree);

  std::vector<index_t> indices;
  valuesd weights;

  rbf_inequality_fitter fitter(model, points);
  std::tie(indices, weights) = fitter.fit(values, values_lb, values_ub, absolute_tolerance);

  rbf_evaluator<> eval(model, take_rows(points, indices));
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate(points);

  for (index_t i = 0; i < n_points; i++) {
    if (!std::isnan(values(i))) {
      EXPECT_LT(std::abs(values_fit(i) - values(i)), absolute_tolerance);
    } else {
      if (!std::isnan(values_lb(i))) {
        EXPECT_GT(values_fit(i), values_lb(i) - absolute_tolerance);
      }
      if (!std::isnan(values_ub(i))) {
        EXPECT_LT(values_fit(i), values_ub(i) + absolute_tolerance);
      }
    }
  }
}
