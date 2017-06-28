// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <map>

#include "rmt_node.hpp"
#include "types.hpp"

namespace polatory {
namespace isosurface {

namespace {

// Coefficients for the three primitive vectors
// to reproduce each NeighborVectors.
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

} // namespace


class rmt_node_list : std::map<cell_index, rmt_node> {
   typedef rmt_node Node;
   typedef std::map<cell_index, Node> base;

   std::array<cell_index_difference, 14> NeighborCellIndexDeltas;

public:
   using iterator = base::iterator;

   using base::at;
   using base::begin;
   using base::clear;
   using base::count;
   using base::end;
   using base::erase;
   using base::find;
   using base::insert;
   using base::size;

   Node *node_ptr(cell_index cell_index)
   {
      auto it = find(cell_index);
      return it != end() ? &it->second : nullptr;
   }

   void init_strides(cell_index_difference stride1, cell_index_difference stride2)
   {
      for (edge_index ei = 0; ei < 14; ei++) {
         auto delta_m = NeighborCellVectors[ei];
         NeighborCellIndexDeltas[ei] =
            delta_m[2] * stride2 + delta_m[1] * stride1 + delta_m[0];
      }
   }

   cell_index neighbor_cell_index(cell_index cell_index, edge_index ei) const
   {
      return static_cast<cell_index_difference>(cell_index) + NeighborCellIndexDeltas[ei];
   }

   iterator find_neighbor_node(cell_index cell_index, edge_index ei)
   {
      return find(neighbor_cell_index(cell_index, ei));
   }

   Node *neighbor_node_ptr(cell_index cell_index, edge_index ei)
   {
      return node_ptr(neighbor_cell_index(cell_index, ei));
   }
};

} // namespace isosurface
} // namespace polatory
