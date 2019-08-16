// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <map>

#include <polatory/isosurface/rmt_node.hpp>
#include <polatory/isosurface/types.hpp>

namespace polatory {
namespace isosurface {

// Coefficients for the three primitive vectors
// to reproduce each NeighborVectors.
extern const std::array<cell_vector, 14> NeighborCellVectors;

class rmt_node_list : std::map<cell_index, rmt_node> {
  using base_type = std::map<cell_index, rmt_node>;

  std::array<cell_index, 14> NeighborCellIndexDeltas;

public:
  using iterator = base_type::iterator;

  using base_type::at;
  using base_type::begin;
  using base_type::clear;
  using base_type::count;
  using base_type::end;
  using base_type::erase;
  using base_type::find;
  using base_type::insert;
  using base_type::size;

  rmt_node *node_ptr(cell_index cell_index) {
    auto it = find(cell_index);
    return it != end() ? &it->second : nullptr;
  }

  void init_strides(cell_index stride1, cell_index stride2) {
    for (edge_index ei = 0; ei < 14; ei++) {
      auto delta_m = NeighborCellVectors[ei];
      NeighborCellIndexDeltas[ei] =
        delta_m(2) * stride2 + delta_m(1) * stride1 + delta_m(0);
    }
  }

  cell_index neighbor_cell_index(cell_index cell_index, edge_index ei) const {
    return cell_index + NeighborCellIndexDeltas[ei];
  }

  iterator find_neighbor_node(cell_index cell_index, edge_index ei) {
    return find(neighbor_cell_index(cell_index, ei));
  }

  rmt_node *neighbor_node_ptr(cell_index cell_index, edge_index ei) {
    return node_ptr(neighbor_cell_index(cell_index, ei));
  }
};

}  // namespace isosurface
}  // namespace polatory
