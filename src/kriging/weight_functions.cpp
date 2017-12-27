// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/kriging/weight_functions.hpp>

#include <cmath>

namespace polatory {
namespace kriging {

const rbf::weight_function weight_functions::n_pairs =
  [](size_t np, double d, double model_g) { return std::sqrt(np); };

const rbf::weight_function weight_functions::n_pairs_over_distance_squared =
  [](size_t np, double d, double model_g) { return std::sqrt(np) / std::abs(d); };

const rbf::weight_function weight_functions::n_pairs_over_model_gamma_squared =
  [](size_t np, double d, double model_g) { return std::sqrt(np) / std::abs(model_g); };

const rbf::weight_function weight_functions::one =
  [](size_t np, double d, double model_g) { return 1.0; };

const rbf::weight_function weight_functions::one_over_distance_squared =
  [](size_t np, double d, double model_g) { return 1.0 / std::abs(d); };

const rbf::weight_function weight_functions::one_over_model_gamma_squared =
  [](size_t np, double d, double model_g) { return 1.0 / std::abs(model_g); };

}  // namespace kriging
}  // namespace polatory
