// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace kriging {

struct weight_functions {
  static const rbf::weight_function n_pairs;
  static const rbf::weight_function n_pairs_over_distance_squared;
  static const rbf::weight_function n_pairs_over_model_gamma_squared;
  static const rbf::weight_function one;
  static const rbf::weight_function one_over_distance_squared;
  static const rbf::weight_function one_over_model_gamma_squared;
};

}  // namespace kriging
}  // namespace polatory
