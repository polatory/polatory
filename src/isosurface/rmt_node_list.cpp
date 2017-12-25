// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/isosurface/rmt_node_list.hpp>

namespace polatory {
namespace isosurface {

const std::array<cell_vector, 14> NeighborCellVectors
  {
    cell_vector(+1, +1, +1),
    cell_vector(+1, +1, +0),
    cell_vector(+0, +0, -1),
    cell_vector(+1, +0, +1),
    cell_vector(+1, +0, +0),
    cell_vector(+0, -1, -1),
    cell_vector(+0, -1, +0),
    cell_vector(-1, -1, -1),
    cell_vector(-1, -1, +0),
    cell_vector(+0, +0, +1),
    cell_vector(-1, +0, -1),
    cell_vector(-1, +0, +0),
    cell_vector(+0, +1, +1),
    cell_vector(+0, +1, +0)
  };

}  // namespace isosurface
}  // namespace polatory
