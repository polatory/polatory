#pragma once

#include <Eigen/Core>
#include <array>
#include <cstdint>

namespace polatory::isosurface {

using cell_index = int64_t;

using cell_vector = Eigen::Vector3i;

using cell_vectors = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;

using vertex_index = int64_t;

using face = std::array<vertex_index, 3>;

}  // namespace polatory::isosurface
