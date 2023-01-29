#pragma once

#include <algorithm>
#include <cmath>
#include <polatory/types.hpp>

namespace polatory::fmm {

inline int fmm_tree_height(index_t points_estimated) {
  return 2 + std::max(2, static_cast<int>(std::floor(std::log(points_estimated) / std::log(8))));
}

}  // namespace polatory::fmm
