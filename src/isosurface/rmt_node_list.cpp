#include <polatory/isosurface/rmt_node_list.hpp>

namespace polatory::isosurface {

namespace detail {

neighbor_cell_vectors::neighbor_cell_vectors()
    : base{{
          cell_vector(+1, +0, +0),  // 0
          cell_vector(+1, +0, +1),  // 1
          cell_vector(+0, +0, +1),  // 2
          cell_vector(+1, +1, +0),  // 3
          cell_vector(+1, +1, +1),  // 4
          cell_vector(+0, +1, +1),  // 5
          cell_vector(+0, +1, +0),  // 6
          cell_vector(-1, +0, +0),  // 7
          cell_vector(-1, +0, -1),  // 8
          cell_vector(+0, +0, -1),  // 9
          cell_vector(-1, -1, +0),  // A
          cell_vector(-1, -1, -1),  // B
          cell_vector(+0, -1, -1),  // C
          cell_vector(+0, -1, +0),  // D
      }} {}

}  // namespace detail

// NOLINTNEXTLINE(cert-err58-cpp)
const detail::neighbor_cell_vectors NeighborCellVectors;

}  // namespace polatory::isosurface
