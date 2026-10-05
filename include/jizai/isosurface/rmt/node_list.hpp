#pragma once

#include <jizai/isosurface/rmt/lattice_coordinates.hpp>
#include <jizai/isosurface/rmt/node.hpp>
#include <unordered_map>

namespace jizai::isosurface::rmt {

class NodeList : public std::unordered_map<LatticeCoordinates, Node, LatticeCoordinatesHash> {
 public:
  Node* node_ptr(const LatticeCoordinates& lc) {
    auto it = find(lc);
    return it != end() ? &it->second : nullptr;
  }
};

}  // namespace jizai::isosurface::rmt
