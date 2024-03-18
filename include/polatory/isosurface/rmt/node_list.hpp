#pragma once

#include <array>
#include <polatory/isosurface/rmt/node.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <unordered_map>

namespace polatory::isosurface::rmt {

// Coefficients for the three primitive vectors
// to reproduce each NeighborVectors.
inline const std::array<cell_vector, 14> kNeighborCellVectors{{
    cell_vector(1, 0, 0),     // 0
    cell_vector(1, 0, 1),     // 1
    cell_vector(0, 0, 1),     // 2
    cell_vector(1, 1, 0),     // 3
    cell_vector(1, 1, 1),     // 4
    cell_vector(0, 1, 1),     // 5
    cell_vector(0, 1, 0),     // 6
    cell_vector(-1, 0, 0),    // 7
    cell_vector(-1, 0, -1),   // 8
    cell_vector(0, 0, -1),    // 9
    cell_vector(-1, -1, 0),   // A
    cell_vector(-1, -1, -1),  // B
    cell_vector(0, -1, -1),   // C
    cell_vector(0, -1, 0),    // D
}};

class node_list : std::unordered_map<cell_vector, node, cell_vector_hash> {
  using Base = std::unordered_map<cell_vector, node, cell_vector_hash>;
  using Node = node;

 public:
  using Base::at;
  using Base::begin;
  using Base::clear;
  using Base::contains;
  using Base::emplace;
  using Base::end;
  using Base::erase;
  using Base::find;
  using Base::size;

  Node* neighbor_node_ptr(const cell_vector& cv, edge_index ei) {
    return node_ptr(cv + kNeighborCellVectors.at(ei));
  }

  Node* node_ptr(const cell_vector& cv) {
    auto it = find(cv);
    return it != end() ? &it->second : nullptr;
  }
};

}  // namespace polatory::isosurface::rmt
