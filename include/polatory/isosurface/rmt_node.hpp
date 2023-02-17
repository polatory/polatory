#pragma once

#include <array>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/bit.hpp>
#include <polatory/isosurface/types.hpp>
#include <unordered_map>
#include <vector>

namespace polatory::isosurface {

// Encodes 0 or 1 on 14 outgoing halfedges for each node.
using edge_bitset = std::uint16_t;

static constexpr edge_bitset EdgeSetMask = 0x3fff;

// Edge index per node: 0 - 13
using edge_index = int;

// Adjacent edges (4 or 6) of each edge.
extern const std::array<edge_bitset, 14> NeighborMasks;

enum binary_sign { Pos = 0, Neg = 1 };

class rmt_node {
  geometry::point3d pos;
  double val{};

 public:
  bool evaluated{};

  // The corresponding bit is set if an edge crosses the isosurface
  // at a point nearer than the midpoint.
  // Such intersections are called "near intersections".
  edge_bitset intersections{};

  // The corresponding bit is set if an edge crosses the isosurface.
  edge_bitset all_intersections{};

  std::unique_ptr<std::vector<vertex_index>> vis;

 private:
  std::unique_ptr<std::array<rmt_node*, 14>> neighbors_;

  static std::vector<edge_bitset> get_holes(edge_bitset edge_set) {
    return get_surfaces(edge_set ^ EdgeSetMask);
  }

  static std::vector<edge_bitset> get_surfaces(edge_bitset edge_set) {
    std::vector<edge_bitset> surfaces;

    edge_bitset remaining_edges = edge_set;

    while (remaining_edges != 0) {
      // visit a new surface
      edge_bitset to_visit_edges = 1 << bit_peek(remaining_edges);
      edge_bitset visited_edges = 0;

      while (to_visit_edges != 0) {
        // scan to_visit_edges and build its neighbor list
        edge_bitset neighbors = 0;
        do {
          edge_index edge_idx = bit_peek(to_visit_edges);
          edge_bitset visiting = 1 << edge_idx;

          // move current edge from to_visit_edges to visited_edges
          to_visit_edges ^= visiting;
          visited_edges |= visiting;

          edge_bitset next = propagate(visiting) & edge_set;
          edge_bitset after_next = propagate(next) & edge_set;
          neighbors |= next & after_next;
        } while (to_visit_edges != 0);

        // update to_visit_edges
        to_visit_edges = neighbors & (~visited_edges & remaining_edges);
      }

      remaining_edges ^= visited_edges;
      surfaces.push_back(visited_edges);
    }

    return surfaces;
  }

  static edge_bitset propagate(edge_bitset edge_set) {
    if (edge_set == 0) {
      return 0;
    }

    edge_bitset neighbors = 0;
    do {
      edge_index edge_idx = bit_pop(&edge_set);
      neighbors |= NeighborMasks.at(edge_idx);
    } while (edge_set != 0);

    return neighbors;
  }

 public:
  explicit rmt_node(const geometry::point3d& position) : pos(position) {}

  void cluster(std::vector<geometry::point3d>& vertices,
               std::unordered_map<vertex_index, vertex_index>& cluster_map) const {
    auto surfaces = get_surfaces(intersections);
    for (auto surface : surfaces) {
      auto holes = get_holes(surface);
      if (holes.size() != 1) {
        continue;
      }

      auto n = bit_count(surface);
      auto new_vi = static_cast<vertex_index>(vertices.size());

      geometry::point3d clustered = geometry::point3d::Zero();
      while (surface != 0) {
        auto edge_idx = bit_pop(&surface);
        auto vi = vertex_on_edge(edge_idx);
        clustered += vertices.at(vi);
        cluster_map.emplace(vi, new_vi);
      }
      clustered /= static_cast<double>(n);

      vertices.push_back(clustered);
    }
  }

  bool has_intersection(edge_index edge_idx) const {
    edge_bitset edge_bit = 1 << edge_idx;
    return (intersections & edge_bit) != 0;
  }

  bool has_neighbor(edge_index edge) const;

  void insert_vertex(vertex_index vi, edge_index edge_idx) {
    POLATORY_ASSERT(!has_intersection(edge_idx));

    if (!vis) {
      vis = std::make_unique<std::vector<vertex_index>>();
    }

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;

    intersections |= edge_bit;

    auto it = vis->begin() + bit_count(static_cast<edge_bitset>(intersections & edge_count_mask));
    vis->insert(it, vi);

    POLATORY_ASSERT(vertex_on_edge(edge_idx) == vi);
  }

  rmt_node& neighbor(edge_index edge);

  const rmt_node& neighbor(edge_index edge) const;

  const geometry::point3d& position() const { return pos; }

  void set_intersection(edge_index edge_idx) {
    edge_bitset edge_bit = 1 << edge_idx;

    all_intersections |= edge_bit;
  }

  void set_neighbors(std::unique_ptr<std::array<rmt_node*, 14>> neighbors) {
    neighbors_.swap(neighbors);
  }

  void set_value(double value) {
    POLATORY_ASSERT(!evaluated);
    this->val = value;
    evaluated = true;
  }

  double value() const {
    POLATORY_ASSERT(evaluated);
    return val;
  }

  binary_sign value_sign() const { return value() < 0 ? Neg : Pos; }

  vertex_index vertex_on_edge(edge_index edge_idx) const {
    POLATORY_ASSERT(has_intersection(edge_idx));

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;
    return vis->at(bit_count(static_cast<edge_bitset>(intersections & edge_count_mask)));
  }
};

}  // namespace polatory::isosurface
