#pragma once

#include <polatory/types.hpp>

namespace polatory::kriging {

class weight_function {
 public:
  static const weight_function num_pairs;
  static const weight_function num_pairs_over_distance_squared;
  static const weight_function num_pairs_over_model_gamma_squared;
  static const weight_function one;
  static const weight_function one_over_distance_squared;
  static const weight_function one_over_model_gamma_squared;

  weight_function(double exp_distance, double exp_model_gamma, double exp_num_pairs)
      : exp_distance_(exp_distance),
        exp_model_gamma_(exp_model_gamma),
        exp_num_pairs_(exp_num_pairs) {}

  double operator()(double distance, double model_gamma, index_t num_pairs) const {
    return std::pow(distance, 0.5 * exp_distance_) * std::pow(model_gamma, 0.5 * exp_model_gamma_) *
           std::pow(static_cast<double>(num_pairs), 0.5 * exp_num_pairs_);
  }

 private:
  double exp_distance_;
  double exp_model_gamma_;
  double exp_num_pairs_;
};

}  // namespace polatory::kriging
