#pragma once

#include <functional>
#include <polatory/types.hpp>

namespace polatory::kriging {

using weight_function = std::function<double(index_t, double, double)>;

struct weight_functions {
  static const weight_function num_pairs;
  static const weight_function num_pairs_over_distance_squared;
  static const weight_function num_pairs_over_model_gamma_squared;
  static const weight_function one;
  static const weight_function one_over_distance_squared;
  static const weight_function one_over_model_gamma_squared;
};

}  // namespace polatory::kriging
