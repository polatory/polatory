#pragma once

#include <array>
#include <polatory/types.hpp>

namespace polatory::isosurface {

using vertex_index = index_t;
using face = std::array<vertex_index, 3>;

}  // namespace polatory::isosurface
