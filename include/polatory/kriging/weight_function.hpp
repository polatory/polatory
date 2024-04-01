#pragma once

#include <cmath>
#include <polatory/types.hpp>

namespace polatory::kriging {

class weight_function {
 public:
  static const weight_function kNumPairs;
  static const weight_function kNumPairsOverDistanceSquared;
  static const weight_function kNumPairsOverModelGammaSquared;
  static const weight_function kOne;
  static const weight_function kOneOverDistanceSquared;
  static const weight_function kOneOverModelGammaSquared;

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

inline const weight_function weight_function::kNumPairs{0.0, 0.0, 1.0};
inline const weight_function weight_function::kNumPairsOverDistanceSquared{-2.0, 0.0, 1.0};
inline const weight_function weight_function::kNumPairsOverModelGammaSquared{0.0, -2.0, 1.0};
inline const weight_function weight_function::kOne{0.0, 0.0, 0.0};
inline const weight_function weight_function::kOneOverDistanceSquared{-2.0, 0.0, 0.0};
inline const weight_function weight_function::kOneOverModelGammaSquared{0.0, -2.0, 0.0};

}  // namespace polatory::kriging
