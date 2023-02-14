#pragma once

#include <array>
#include <polatory/isosurface/rmt_node.hpp>
#include <polatory/isosurface/types.hpp>
#include <unordered_map>

namespace polatory::isosurface {

namespace detail {

class neighbor_cell_vectors : public std::array<cell_vector, 14> {
  using base = std::array<cell_vector, 14>;

 public:
  neighbor_cell_vectors();
};

}  // namespace detail

// Coefficients for the three primitive vectors
// to reproduce each NeighborVectors.
extern const detail::neighbor_cell_vectors NeighborCellVectors;

class rmt_node_list : std::unordered_map<cell_vector, rmt_node> {
  using base = std::unordered_map<cell_vector, rmt_node>;

 public:
  using base::at;
  using base::begin;
  using base::clear;
  using base::contains;
  using base::emplace;
  using base::end;
  using base::erase;
  using base::find;
  using base::size;

  rmt_node* node_ptr(const cell_vector& cv) {
    auto it = find(cv);
    return it != end() ? &it->second : nullptr;
  }

  rmt_node* neighbor_node_ptr(const cell_vector& cv, edge_index ei) {
    return node_ptr(cv + NeighborCellVectors.at(ei));
  }
};

}  // namespace polatory::isosurface
