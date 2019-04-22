// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <functional>

namespace polatory {
namespace kriging {

using weight_function = std::function<double(size_t, double, double)>;

struct weight_functions {
  static const weight_function n_pairs;
  static const weight_function n_pairs_over_distance_squared;
  static const weight_function n_pairs_over_model_gamma_squared;
  static const weight_function one;
  static const weight_function one_over_distance_squared;
  static const weight_function one_over_model_gamma_squared;
};

}  // namespace kriging
}  // namespace polatory
