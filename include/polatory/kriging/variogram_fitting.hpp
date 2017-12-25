// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <vector>

#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace kriging {

static const rbf::weight_function weight_n =
  [](size_t n, double h, double model_gamma) { return std::sqrt(n); };

static const rbf::weight_function weight_n_over_gamma_squared =
  [](size_t n, double h, double model_gamma) { return std::sqrt(n) / std::abs(model_gamma); };

static const rbf::weight_function weight_n_over_h_squared =
  [](size_t n, double h, double model_gamma) { return std::sqrt(n) / std::abs(h); };

static const rbf::weight_function weight_one =
  [](size_t n, double h, double model_gamma) { return 1.0; };

static const rbf::weight_function weight_one_over_gamma_squared =
  [](size_t n, double h, double model_gamma) { return 1.0 / std::abs(model_gamma); };

static const rbf::weight_function weight_one_over_h_squared =
  [](size_t n, double h, double model_gamma) { return 1.0 / std::abs(h); };

class variogram_fitting {
public:
  variogram_fitting(const empirical_variogram& emp_variog, const rbf::rbf& rbf,
                    rbf::weight_function weight = weight_one);

  const std::vector<double>& parameters() const;

private:
  std::vector<double> params_;
};

}  // namespace kriging
}  // namespace polatory
