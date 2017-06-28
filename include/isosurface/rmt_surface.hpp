// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <cassert>
#include <vector>

#include <boost/operators.hpp>

#include "../common/uncertain.hpp"
#include "rmt_lattice.hpp"
#include "rmt_node.hpp"
#include "rmt_node_list.hpp"
#include "types.hpp"

namespace polatory {
namespace isosurface {

namespace detail {

class rmt_tetrahedron {
   friend class rmt_tetrahedron_iterator;

   // List of indices of the three edges of each tetrahedron.
   static constexpr std::array<std::array<edge_index, 3>, 6> EdgeIndices
   { {
      { 0, 1, 4 },
      { 0, 4, 3 },
      { 0, 3, 9 },
      { 0, 9, 12 },
      { 0, 12, 13 },
      { 0, 13, 1 }
   } };

   // List of indices of the three outer edges of each tetrahedron as:
   //          ei0           ei1           ei2
   //  node0 ------> node1 ------> node2 ------> node0
   //        <------       <------       <------ 
   //         ~ei0          ~ei1          ~ei2
   // where
   //  ei0: outer edge from node0 to node1
   //  ~ei0: opposite outer edge of e0 from node1 to node0.
   static constexpr std::array<std::array<edge_index, 3>, 6> OuterEdgeIndices
   { {
      { 2, 6, 12 },
      { 5, 9, 13 },
      { 6, 11, 1 },
      { 8, 13, 4 },
      { 11, 2, 3 },
      { 10, 4, 9 }
   } };

   // Encode four signs of tetrahedron nodes into an integer.
   template <binary_sign s, binary_sign s0, binary_sign s1, binary_sign s2>
   static constexpr int tetrahedron_type()
   {
      return (s2 << 3) | (s1 << 2) | (s0 << 1) | s;
   }

   static int tetrahedron_type(binary_sign s, binary_sign s0, binary_sign s1, binary_sign s2)
   {
      return (s2 << 3) | (s1 << 2) | (s0 << 1) | s;
   }

   static common::uncertain<vertex_index> vertex_on_edge(const rmt_node& node, edge_index edge_idx, const rmt_node& opp_node)
   {
      if (node.has_intersection(edge_idx))
         return node.vertex_on_edge(edge_idx);

      auto opp_edge_idx = OppositeEdge[edge_idx];
      if (opp_node.has_intersection(opp_edge_idx))
         return opp_node.vertex_on_edge(opp_edge_idx);

      return common::uncertain<vertex_index>();
   }

   const rmt_node& node;
   const rmt_node_list& node_list;
   const int index;

public:
   rmt_tetrahedron(const rmt_node& node, const rmt_node_list& node_list, int index)
      : node(node)
      , node_list(node_list)
      , index(index)
   {
   }

   // Returns 0, 1 or 2 triangular faces of the isosurface in this tetrahedron.
   std::vector<face> get_faces() const
   {
      auto ei0 = EdgeIndices[index][0];
      auto ei1 = EdgeIndices[index][1];
      auto ei2 = EdgeIndices[index][2];

      auto oei0 = OuterEdgeIndices[index][0];
      auto oei1 = OuterEdgeIndices[index][1];
      auto oei2 = OuterEdgeIndices[index][2];

      const auto& node0 = *node.neighbor_cache[ei0];
      const auto& node1 = *node.neighbor_cache[ei1];
      const auto& node2 = *node.neighbor_cache[ei2];

      // Check six edges to obtain vertices

      auto v0 = vertex_on_edge(node, ei0, node0);
      auto v1 = vertex_on_edge(node, ei1, node1);
      auto v2 = vertex_on_edge(node, ei2, node2);
      auto v3 = vertex_on_edge(node0, oei0, node1);
      auto v4 = vertex_on_edge(node1, oei1, node2);
      auto v5 = vertex_on_edge(node2, oei2, node0);

      auto tetra = tetrahedron_type(node.value_sign(), node0.value_sign(), node1.value_sign(), node2.value_sign());

      std::vector<face> faces;

      switch (tetra) {
      case tetrahedron_type<Pos, Pos, Pos, Pos>() :
         // no faces.
         break;
      case tetrahedron_type<Neg, Neg, Neg, Neg>() :
         // no faces.
         break;
      case tetrahedron_type<Neg, Pos, Pos, Pos>() :
         // v0-v1-v2
         faces.push_back({ v0.get(), v1.get(), v2.get() });
         break;
      case tetrahedron_type<Pos, Neg, Neg, Neg>() :
         // v0-v2-v1
         faces.push_back({ v0.get(), v2.get(), v1.get() });
         break;
      case tetrahedron_type<Pos, Neg, Pos, Pos>() :
         // v0-v5-v3
         faces.push_back({ v0.get(), v5.get(), v3.get() });
         break;
      case tetrahedron_type<Neg, Pos, Neg, Neg>() :
         // v0-v3-v5
         faces.push_back({ v0.get(), v3.get(), v5.get() });
         break;
      case tetrahedron_type<Pos, Pos, Neg, Pos>() :
         // v1-v3-v4
         faces.push_back({ v1.get(), v3.get(), v4.get() });
         break;
      case tetrahedron_type<Neg, Neg, Pos, Neg>() :
         // v1-v4-v3
         faces.push_back({ v1.get(), v4.get(), v3.get() });
         break;
      case tetrahedron_type<Pos, Pos, Pos, Neg>() :
         // v2-v4-v5
         faces.push_back({ v2.get(), v4.get(), v5.get() });
         break;
      case tetrahedron_type<Neg, Neg, Neg, Pos>() :
         // v2-v5-v4
         faces.push_back({ v2.get(), v5.get(), v4.get() });
         break;
      case tetrahedron_type<Neg, Neg, Pos, Pos>() :
         // v5-v3-v1, v5-v1-v2
         faces.push_back({ v5.get(), v3.get(), v1.get() });
         faces.push_back({ v5.get(), v1.get(), v2.get() });
         break;
      case tetrahedron_type<Pos, Pos, Neg, Neg>() :
         // v5-v1-v3, v5-v2-v1
         faces.push_back({ v5.get(), v1.get(), v3.get() });
         faces.push_back({ v5.get(), v2.get(), v1.get() });
         break;
      case tetrahedron_type<Neg, Pos, Neg, Pos>() :
         // v0-v3-v4, v0-v4-v2
         faces.push_back({ v0.get(), v3.get(), v4.get() });
         faces.push_back({ v0.get(), v4.get(), v2.get() });
         break;
      case tetrahedron_type<Pos, Neg, Pos, Neg>() :
         // v0-v4-v3, v0-v2-v4
         faces.push_back({ v0.get(), v4.get(), v3.get() });
         faces.push_back({ v0.get(), v2.get(), v4.get() });
         break;
      case tetrahedron_type<Neg, Pos, Pos, Neg>() :
         // v5-v0-v1, v5-v1-v4
         faces.push_back({ v5.get(), v0.get(), v1.get() });
         faces.push_back({ v5.get(), v1.get(), v4.get() });
         break;
      case tetrahedron_type<Pos, Neg, Neg, Pos>() :
         // v5-v1-v0, v5-v4-v1
         faces.push_back({ v5.get(), v1.get(), v0.get() });
         faces.push_back({ v5.get(), v4.get(), v1.get() });
         break;
      default:
         assert(false);
         break;
      }

      return faces;
   }
};

class rmt_tetrahedron_iterator
   : public boost::input_iterator_helper<rmt_tetrahedron_iterator, rmt_tetrahedron, int> {
   const rmt_node& node;
   const rmt_node_list& node_list;
   int index;

   // Returns if all tetrahedron nodes exist.
   bool tetrahedron_exists() const
   {
      return
         node.neighbor_cache[rmt_tetrahedron::EdgeIndices[index][0]] != nullptr &&
         node.neighbor_cache[rmt_tetrahedron::EdgeIndices[index][1]] != nullptr &&
         node.neighbor_cache[rmt_tetrahedron::EdgeIndices[index][2]] != nullptr;
   }

   // The number of tetrahedra in each cell.
   static constexpr int number_of_tetrahedra = 6;

public:
   rmt_tetrahedron_iterator(const rmt_node& node, const rmt_node_list& node_list)
      : node(node)
      , node_list(node_list)
      , index(0)
   {
      while (is_valid() && !tetrahedron_exists()) {
         // Some of the tetrahedron nodes do not exist.
         index++;
      }
   }

   bool operator==(const rmt_tetrahedron_iterator& other) const
   {
      return index == other.index;
   }

   rmt_tetrahedron_iterator& operator++()
   {
      if (!is_valid()) {
         assert(false);
         return *this;
      }

      do {
         index++;
      } while (is_valid() && !tetrahedron_exists());

      return *this;
   }

   rmt_tetrahedron operator*() const
   {
      return rmt_tetrahedron(node, node_list, index);
   }

   bool is_valid() const
   {
      return index < number_of_tetrahedra;
   }
};

} // namespace detail

class rmt_surface {
   const rmt_lattice& lattice;
   std::vector<face> faces;

   void add_face(const face& face)
   {
      auto v0 = lattice.clustered_vertex_index(face[0]);
      auto v1 = lattice.clustered_vertex_index(face[1]);
      auto v2 = lattice.clustered_vertex_index(face[2]);
      if (v0 == v1 || v1 == v2 || v2 == v0)
         return;

      faces.push_back({ v0, v1, v2 });
   }
   
public:
   explicit rmt_surface(const rmt_lattice& lattice)
      : lattice(lattice)
   {
   }

   const std::vector<face>& get_faces() const
   {
      return faces;
   }

   void generate_surface()
   {
      faces.clear();

      for (const auto& nodei : lattice.node_list) {
         auto& node = nodei.second;
         
         for (detail::rmt_tetrahedron_iterator it(node, lattice.node_list); it.is_valid(); ++it) {
            auto tetra = *it;
            
            for (const auto& face : tetra.get_faces()) {
               add_face(face);
            }
         }
      }
   }
};

} // namespace isosurface
} // namespace polatory
