#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_inequality_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/precision.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <utility>

#include "../utility.hpp"

using polatory::index_t;
using polatory::model;
using polatory::precision;
using polatory::common::valuesd;
using polatory::geometry::points1d;
using polatory::interpolation::rbf_evaluator;
using polatory::interpolation::rbf_inequality_fitter;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;

TEST(rbf_inequality_fitter, inequality_only) {
  constexpr int kDim = 3;

  index_t n_points = 10000;
  auto absolute_tolerance = 1e-4;

  auto aniso = random_anisotropy<kDim>();
  auto [points, values] = sample_data(n_points, aniso);

  valuesd values_lb = values.array() - 0.001;
  valuesd values_ub = values.array() + 0.001;
  values = valuesd::Constant(n_points, std::numeric_limits<double>::quiet_NaN());

  biharmonic3d<kDim> rbf({1.0});
  rbf.set_anisotropy(aniso);

  auto poly_degree = rbf.cpd_order() - 1;
  model<kDim> model(std::move(rbf), poly_degree);

  rbf_inequality_fitter<kDim> fitter(model, points);
  auto [indices, weights] = fitter.fit(values, values_lb, values_ub, absolute_tolerance, 32);

  EXPECT_EQ(weights.rows(), indices.size() + model.poly_basis_size());

  rbf_evaluator<kDim> eval(model, points(indices, Eigen::all), precision::kPrecise);
  eval.set_weights(weights);
  valuesd values_fit = eval.evaluate(points);

  for (index_t i = 0; i < n_points; i++) {
    EXPECT_GT(values_fit(i), values_lb(i) - absolute_tolerance);
    EXPECT_LT(values_fit(i), values_ub(i) + absolute_tolerance);
  }
}

// Example problem taken from https://doi.org/10.1007/BF00897655
TEST(rbf_inequality_fitter, kostov86) {
  constexpr int kDim = 1;

  index_t n_points = 25;
  const auto absolute_tolerance = 1e-5;

  points1d points = points1d::Zero(n_points, kDim);
  for (index_t i = 0; i < n_points; i++) {
    points(i, 0) = static_cast<double>(i);
  }

  auto nan = std::numeric_limits<double>::quiet_NaN();
  valuesd values(n_points);
  values << 1, nan, 4, nan, 3, nan, nan, nan, nan, nan, 3, nan, nan, 7, nan, 8, nan, nan, nan, 8, 3,
      nan, nan, 6, nan;

  valuesd values_lb(n_points);
  values_lb << nan, 6, nan, 2, nan, 2, 9, 4, 3, 3, nan, nan, 7, nan, 4, nan, nan, nan, 5, nan, nan,
      9, 5, nan, nan;

  valuesd values_ub(n_points);
  values_ub << nan, nan, nan, 4, nan, 4, nan, nan, nan, nan, nan, 1, nan, nan, nan, nan, 4, 7, nan,
      nan, nan, nan, nan, nan, 3;

  cov_exponential<kDim> rbf({1.0, 3.0});
  model<kDim> model(std::move(rbf), -1);

  rbf_inequality_fitter<kDim> fitter(model, points);
  auto [indices, weights] = fitter.fit(values, values_lb, values_ub, absolute_tolerance, 32);

  rbf_evaluator<kDim> eval(model, points(indices, Eigen::all), precision::kPrecise);
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
