// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <cmath>

namespace polatory {
namespace fmm {

inline
int fmm_tree_height(size_t points_estimated) {
  return 2 + std::max(2, static_cast<int>(std::floor(std::log(points_estimated) / std::log(8))));
}

} // namespace fmm
} // namespace polatory
