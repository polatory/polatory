#pragma once

namespace jizai::isosurface {

enum class BinarySign { kPos, kNeg };

inline BinarySign sign(double x) { return x < 0.0 ? BinarySign::kNeg : BinarySign::kPos; }

}  // namespace jizai::isosurface
