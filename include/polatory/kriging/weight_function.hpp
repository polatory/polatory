#pragma once

#include <cmath>
#include <polatory/types.hpp>

namespace polatory::kriging {

class WeightFunction {
 public:
  static const WeightFunction kNumPairs;
  static const WeightFunction kNumPairsOverDistanceSquared;
  static const WeightFunction kNumPairsOverModelGammaSquared;
  static const WeightFunction kOne;
  static const WeightFunction kOneOverDistanceSquared;
  static const WeightFunction kOneOverModelGammaSquared;

  WeightFunction(double exp_distance, double exp_model_gamma, double exp_num_pairs)
      : exp_distance_(exp_distance),
        exp_model_gamma_(exp_model_gamma),
        exp_num_pairs_(exp_num_pairs) {}

  double operator()(double distance, double model_gamma, Index num_pairs) const {
    return std::pow(distance, 0.5 * exp_distance_) * std::pow(model_gamma, 0.5 * exp_model_gamma_) *
           std::pow(static_cast<double>(num_pairs), 0.5 * exp_num_pairs_);
  }

 private:
  double exp_distance_;
  double exp_model_gamma_;
  double exp_num_pairs_;
};

inline const WeightFunction WeightFunction::kNumPairs{0.0, 0.0, 1.0};
inline const WeightFunction WeightFunction::kNumPairsOverDistanceSquared{-2.0, 0.0, 1.0};
inline const WeightFunction WeightFunction::kNumPairsOverModelGammaSquared{0.0, -2.0, 1.0};
inline const WeightFunction WeightFunction::kOne{0.0, 0.0, 0.0};
inline const WeightFunction WeightFunction::kOneOverDistanceSquared{-2.0, 0.0, 0.0};
inline const WeightFunction WeightFunction::kOneOverModelGammaSquared{0.0, -2.0, 0.0};

}  // namespace polatory::kriging
