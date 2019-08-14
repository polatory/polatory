// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/kriging/weight_function.hpp>

#include <cmath>

namespace polatory {
namespace kriging {

const weight_function weight_functions::n_pairs =
  [](index_t np, double, double) { return std::sqrt(np); };

const weight_function weight_functions::n_pairs_over_distance_squared =
  [](index_t np, double d, double) { return std::sqrt(np) / std::abs(d); };

const weight_function weight_functions::n_pairs_over_model_gamma_squared =
  [](index_t np, double, double model_g) { return std::sqrt(np) / std::abs(model_g); };

const weight_function weight_functions::one =
  [](index_t, double, double) { return 1.0; };

const weight_function weight_functions::one_over_distance_squared =
  [](index_t, double d, double) { return 1.0 / std::abs(d); };

const weight_function weight_functions::one_over_model_gamma_squared =
  [](index_t, double, double model_g) { return 1.0 / std::abs(model_g); };

}  // namespace kriging
}  // namespace polatory
