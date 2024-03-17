#include <polatory/isosurface/rmt_node_list.hpp>

namespace polatory::isosurface {

namespace detail {

neighbor_cell_vectors::neighbor_cell_vectors()
    : base{{
          cell_vector(+0, +0, +1),  // 0
          cell_vector(+1, +0, +1),  // 1
          cell_vector(+1, +0, +0),  // 2
          cell_vector(+0, +1, +1),  // 3
          cell_vector(+1, +1, +1),  // 4
          cell_vector(+1, +1, +0),  // 5
          cell_vector(+0, +1, +0),  // 6
          cell_vector(+0, +0, -1),  // 7
          cell_vector(-1, +0, -1),  // 8
          cell_vector(-1, +0, +0),  // 9
          cell_vector(+0, -1, -1),  // A
          cell_vector(-1, -1, -1),  // B
          cell_vector(-1, -1, +0),  // C
          cell_vector(+0, -1, +0),  // D
      }} {}

}  // namespace detail

// NOLINTNEXTLINE(cert-err58-cpp)
const detail::neighbor_cell_vectors NeighborCellVectors;

}  // namespace polatory::isosurface
