#include <cmath>
#include <polatory/kriging/weight_function.hpp>

namespace polatory {
namespace kriging {

// In the following definitions, we assume that construction of lambda does not throw.

const weight_function weight_functions::n_pairs(  // NOLINT(cert-err58-cpp)
    [](index_t np, double /*d*/, double /*model_g*/) { return std::sqrt(np); });

const weight_function weight_functions::n_pairs_over_distance_squared(  // NOLINT(cert-err58-cpp)
    [](index_t np, double d, double /*model_g*/) { return std::sqrt(np) / std::abs(d); });

const weight_function weight_functions::n_pairs_over_model_gamma_squared(  // NOLINT(cert-err58-cpp)
    [](index_t np, double /*d*/, double model_g) { return std::sqrt(np) / std::abs(model_g); });

const weight_function weight_functions::one(  // NOLINT(cert-err58-cpp)
    [](index_t /*np*/, double /*d*/, double /*model_g*/) { return 1.0; });

const weight_function weight_functions::one_over_distance_squared(  // NOLINT(cert-err58-cpp)
    [](index_t /*np*/, double d, double /*model_g*/) { return 1.0 / std::abs(d); });

const weight_function weight_functions::one_over_model_gamma_squared(  // NOLINT(cert-err58-cpp)
    [](index_t /*np*/, double /*d*/, double model_g) { return 1.0 / std::abs(model_g); });

}  // namespace kriging
}  // namespace polatory
