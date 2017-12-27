// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/kriging/variogram_fitting.hpp>

#include <iostream>

#include <ceres/ceres.h>

namespace polatory {
namespace kriging {

variogram_fitting::variogram_fitting(const empirical_variogram& emp_variog, const rbf::rbf& rbf,
                                     rbf::weight_function weight_fn) {
  auto& cov = rbf.get();

  auto n_params = cov.num_parameters();
  params_ = cov.parameters();

  auto& bin_distance = emp_variog.bin_distance();
  auto& bin_gamma = emp_variog.bin_gamma();
  auto& bin_num_pairs = emp_variog.bin_num_pairs();
  auto n_bins = bin_num_pairs.size();

  ceres::Problem problem;
  for (size_t i = 0; i < n_bins; i++) {
    if (bin_num_pairs[i] == 0)
      continue;

    auto cost_fn = cov.cost_function(bin_num_pairs[i], bin_distance[i], bin_gamma[i], weight_fn);
    problem.AddResidualBlock(cost_fn, nullptr, params_.data());
  }

  for (size_t i = 0; i < n_params; i++) {
    problem.SetParameterLowerBound(params_.data(), i, cov.parameter_lower_bounds()[i]);
    problem.SetParameterUpperBound(params_.data(), i, cov.parameter_upper_bounds()[i]);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 32;
  options.linear_solver_type = ceres::DENSE_QR;

  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
}

const std::vector<double>& variogram_fitting::parameters() const {
  return params_;
}

}  // namespace kriging
}  // namespace polatory
