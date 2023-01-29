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

class rmt_node_list : std::unordered_map<cell_index, rmt_node> {
  using base = std::unordered_map<cell_index, rmt_node>;

  std::array<cell_index, 14> NeighborCellIndexDeltas{};

 public:
  using base::iterator;

  using base::at;
  using base::begin;
  using base::clear;
  using base::contains;
  using base::emplace;
  using base::end;
  using base::erase;
  using base::find;
  using base::size;

  rmt_node *node_ptr(cell_index ci) {
    auto it = find(ci);
    return it != end() ? &it->second : nullptr;
  }

  void init_strides(cell_index stride1, cell_index stride2) {
    for (edge_index ei = 0; ei < 14; ei++) {
      auto delta_cv = NeighborCellVectors.at(ei);
      NeighborCellIndexDeltas.at(ei) = delta_cv(2) * stride2 + delta_cv(1) * stride1 + delta_cv(0);
    }
  }

  cell_index neighbor_cell_index(cell_index ci, edge_index ei) const {
    return ci + NeighborCellIndexDeltas.at(ei);
  }

  iterator find_neighbor_node(cell_index ci, edge_index ei) {
    return find(neighbor_cell_index(ci, ei));
  }

  rmt_node *neighbor_node_ptr(cell_index ci, edge_index ei) {
    return node_ptr(neighbor_cell_index(ci, ei));
  }
};

}  // namespace polatory::isosurface
