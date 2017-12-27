// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <cstdint>

#include <Eigen/Core>

namespace polatory {
namespace isosurface {

using cell_index = uint64_t;

using cell_index_difference = int64_t;

using cell_vector = Eigen::Vector3i;

using vertex_index = size_t;

using face = std::array<vertex_index, 3>;

}  // namespace isosurface
}  // namespace polatory
