#include <polatory/isosurface/rmt_node_list.hpp>

namespace polatory::isosurface {

namespace detail {

neighbor_cell_vectors::neighbor_cell_vectors()
    : base{{cell_vector(+1, +1, +1), cell_vector(+1, +1, +0), cell_vector(+0, +0, -1),
            cell_vector(+1, +0, +1), cell_vector(+1, +0, +0), cell_vector(+0, -1, -1),
            cell_vector(+0, -1, +0), cell_vector(-1, -1, -1), cell_vector(-1, -1, +0),
            cell_vector(+0, +0, +1), cell_vector(-1, +0, -1), cell_vector(-1, +0, +0),
            cell_vector(+0, +1, +1), cell_vector(+0, +1, +0)}} {}

}  // namespace detail

// NOLINTNEXTLINE(cert-err58-cpp)
const detail::neighbor_cell_vectors NeighborCellVectors;

}  // namespace polatory::isosurface
