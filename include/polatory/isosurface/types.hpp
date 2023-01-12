#pragma once

#include <array>
#include <cstdint>

#include <Eigen/Core>

namespace polatory {
namespace isosurface {

using cell_index = int64_t;

using cell_vector = Eigen::Vector3i;

using cell_vectors = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;

using vertex_index = int64_t;

using face = std::array<vertex_index, 3>;

}  // namespace isosurface
}  // namespace polatory
