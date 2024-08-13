#pragma once

#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/node.hpp>
#include <unordered_map>

namespace polatory::isosurface::rmt {

class NodeList : public std::unordered_map<LatticeCoordinates, Node, LatticeCoordinatesHash> {
 public:
  Node* node_ptr(const LatticeCoordinates& lc) {
    auto it = find(lc);
    return it != end() ? &it->second : nullptr;
  }
};

}  // namespace polatory::isosurface::rmt
