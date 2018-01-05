// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_inequality_fitter.hpp>
#include <polatory/rbf/cov_exponential.hpp>

using polatory::common::take_rows;
using polatory::common::valuesd;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_inequality_fitter;
using polatory::rbf::cov_exponential;

// Example problem taken from
//   Kostov, C. & Dubrule, O. Math Geol (1986) 18: 53. https://doi.org/10.1007/BF00897655
TEST(rbf_inequality_fitter, kostov86) {
  const size_t n_points = 25;
  const int poly_dimension = 1;
  const int poly_degree = -1;
  const double absolute_tolerance = 1e-5;

  points3d points = points3d::Zero(n_points, 3);
  for (size_t i = 0; i < n_points; i++) {
    points(i, 0) = i;
  }

  auto nan = std::numeric_limits<double>::quiet_NaN();
  valuesd values(n_points);
  values <<
    1, nan, 4, nan, 3,
    nan, nan, nan, nan, nan,
    3, nan, nan, 7, nan,
    8, nan, nan, nan, 8,
    3, nan, nan, 6, nan;

  valuesd values_lower(n_points);
  values_lower <<
    nan, 6, nan, 2, nan,
    2, 9, 4, 3, 3,
    nan, nan, 7, nan, 4,
    nan, nan, nan, 5, nan,
    nan, 9, 5, nan, nan;

  valuesd values_upper(n_points);
  values_upper <<
    nan, nan, nan, 4, nan,
    4, nan, nan, nan, nan,
    nan, 1, nan, nan, nan,
    nan, 4, 7, nan, nan,
    nan, nan, nan, nan, 3;

  cov_exponential rbf({ 1.0, 3.0, 0.0 });

  std::vector<size_t> indices;
  valuesd weights;

  rbf_inequality_fitter fitter(rbf, poly_dimension, poly_degree, points);
  std::tie(indices, weights) = fitter.fit(values, values_lower, values_upper, absolute_tolerance);

  rbf_evaluator<> eval(rbf, poly_dimension, poly_degree, take_rows(points, indices));
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate_points(points);

  for (size_t i = 0; i < n_points; i++) {
    if (!std::isnan(values(i))) {
      ASSERT_LT(std::abs(values_fit(i) - values(i)), absolute_tolerance);
    } else {
      if (!std::isnan(values_lower(i))) {
        ASSERT_GT(values_fit(i), values_lower(i) - absolute_tolerance);
      }
      if (!std::isnan(values_upper(i))) {
        ASSERT_LT(values_fit(i), values_upper(i) + absolute_tolerance);
      }
    }
  }
}
