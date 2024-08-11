#pragma once

namespace polatory::isosurface {

enum class binary_sign { kPos, kNeg };

inline binary_sign sign(double x) { return x < 0.0 ? binary_sign::kNeg : binary_sign::kPos; }

}  // namespace polatory::isosurface
