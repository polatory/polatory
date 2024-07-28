#pragma once

#include <polatory/isosurface/rmt/node.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <unordered_map>

namespace polatory::isosurface::rmt {

class node_list : public std::unordered_map<cell_vector, node, cell_vector_hash> {
 public:
  node* node_ptr(const cell_vector& cv) {
    auto it = find(cv);
    return it != end() ? &it->second : nullptr;
  }
};

}  // namespace polatory::isosurface::rmt
