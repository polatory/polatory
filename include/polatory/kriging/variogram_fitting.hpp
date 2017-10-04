// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <iostream>

#include "empirical_variogram.hpp"
#include "polatory/rbf/covariance_function.hpp"
#include "polatory/third_party/ceres/ceres.h"

namespace polatory {
namespace kriging {

enum class variogram_fitting_weights {
  cressie,
  equal,
  npairs
};

class variogram_fitting {
  std::vector<double> params;

public:
  variogram_fitting(const empirical_variogram& emp_variog, const rbf::covariance_function *cov,
                    variogram_fitting_weights weights = variogram_fitting_weights::equal) {
    int n_params = cov->num_parameters();
    params = std::vector<double>(cov->parameters());

    auto n_bins = emp_variog.num_bins();
    auto bin_lags = emp_variog.bin_lags();
    const auto& bin_pairs = emp_variog.bin_num_pairs();
    const auto& bin_variog = emp_variog.bin_variogram();

    ceres::Problem problem;
    for (int i = 0; i < n_bins; i++) {
      double weight;
      ceres::CostFunction *cost_function;

      switch (weights) {
      case variogram_fitting_weights::cressie:
        weight = std::sqrt(bin_pairs[i]);
        cost_function = cov->cost_function_over_gamma(bin_lags[i], bin_variog[i], weight);
        break;

      case variogram_fitting_weights::npairs:
        weight = std::sqrt(bin_pairs[i]);
        cost_function = cov->cost_function(bin_lags[i], bin_variog[i], weight);
        break;

      default:
        weight = 1.0;
        cost_function = cov->cost_function(bin_lags[i], bin_variog[i], weight);
        break;
      }

      problem.AddResidualBlock(cost_function, nullptr, params.data());
    }

    for (int i = 0; i < n_params; i++) {
      problem.SetParameterLowerBound(params.data(), i, cov->parameter_lower_bounds()[i]);
      problem.SetParameterUpperBound(params.data(), i, cov->parameter_upper_bounds()[i]);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 32;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    //options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
  }

  const std::vector<double>& parameters() const {
    return params;
  }
};

} // namespace kriging
} // namespace polatory
