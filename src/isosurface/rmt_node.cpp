#include <polatory/isosurface/rmt_node.hpp>

namespace polatory::isosurface {

const std::array<edge_bitset, 14> NeighborMasks{0x321a, 0x2015, 0x24b2, 0x0251, 0x006f,
                                                0x00d4, 0x03b8, 0x0d64, 0x0ac0, 0x1949,
                                                0x2884, 0x3780, 0x2a01, 0x1c07};

bool rmt_node::has_neighbor(edge_index edge) const { return neighbors_->at(edge) != nullptr; }

rmt_node& rmt_node::neighbor(edge_index edge) {
  POLATORY_ASSERT(has_neighbor(edge));
  return *neighbors_->at(edge);
}

const rmt_node& rmt_node::neighbor(edge_index edge) const {
  POLATORY_ASSERT(has_neighbor(edge));
  return *neighbors_->at(edge);
}

}  // namespace polatory::isosurface
