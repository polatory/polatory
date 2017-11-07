// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

#include "polatory/interpolation/rbf_evaluator.hpp"
#include "polatory/interpolation/rbf_fitter.hpp"
#include "polatory/numeric/sum_accumulator.hpp"
#include "polatory/rbf/rbf_base.hpp"

namespace polatory {
namespace kriging {

class cross_validation {
public:
  static double leave_one_out(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                              const std::vector<Eigen::Vector3d>& points, const Eigen::VectorXd& values,
                              double absolute_tolerance) {
    size_t n_points = points.size();

    std::vector<Eigen::Vector3d> points_one_out(n_points - 1);
    Eigen::VectorXd values_one_out(n_points - 1);

    std::vector<Eigen::Vector3d> test_points(1);

    numeric::kahan_sum_accumulator<double> squared_error;

    for (size_t i = 0; i < n_points; i++) {
      std::copy(points.begin(), points.begin() + i, points_one_out.begin());
      std::copy(points.begin() + i + 1, points.end(), points_one_out.begin() + i);

      values_one_out.head(i) = values.head(i);
      values_one_out.tail(n_points - i - 1) = values.tail(n_points - i - 1);

      interpolation::rbf_fitter fitter(rbf, poly_dimension, poly_degree, points_one_out);
      auto weights = fitter.fit(values_one_out, absolute_tolerance);

      test_points[0] = points[i];

      interpolation::rbf_evaluator<> eval(rbf, poly_dimension, poly_degree, points_one_out);
      eval.set_weights(weights);
      auto values_fit = eval.evaluate_points(test_points);

      squared_error += std::pow(values_fit[0] - values[i], 2.0);
    }

    return squared_error.get() / n_points;
  }
};

} // namespace kriging
} // namespace polatory
