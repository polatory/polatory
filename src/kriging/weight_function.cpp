#include <polatory/kriging/weight_function.hpp>

namespace polatory::kriging {

const weight_function weight_function::num_pairs{0.0, 0.0, 1.0};

const weight_function weight_function::num_pairs_over_distance_squared{-2.0, 0.0, 1.0};

const weight_function weight_function::num_pairs_over_model_gamma_squared{0.0, -2.0, 1.0};

const weight_function weight_function::one{0.0, 0.0, 0.0};

const weight_function weight_function::one_over_distance_squared{-2.0, 0.0, 0.0};

const weight_function weight_function::one_over_model_gamma_squared{0.0, -2.0, 0.0};

}  // namespace polatory::kriging
