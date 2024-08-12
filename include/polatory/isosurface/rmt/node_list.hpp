#pragma once

#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/node.hpp>
#include <unordered_map>

namespace polatory::isosurface::rmt {

class node_list : public std::unordered_map<lattice_coordinates, node, lattice_coordinates_hash> {
 public:
  node* node_ptr(const lattice_coordinates& lc) {
    auto it = find(lc);
    return it != end() ? &it->second : nullptr;
  }
};

}  // namespace polatory::isosurface::rmt
