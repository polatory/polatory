#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/bit.hpp>
#include <polatory/isosurface/types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface {

namespace detail {

class neighbor_edge_pairs : public std::array<std::vector<std::pair<int, int>>, 14> {
  using base = std::array<std::vector<std::pair<int, int>>, 14>;

 public:
  neighbor_edge_pairs();
};

}  // namespace detail

// Encodes 0 or 1 on 14 outgoing halfedges for each node.
using edge_bitset = std::uint16_t;
// Encodes 0 or 1 on 24 faces for each node.
using face_bitset = std::uint32_t;

static constexpr face_bitset FaceSetMask = 0xffffff;

// Edge index per node: 0 - 13
using edge_index = int;
// 0 - 23
using face_index = int;

// Adjacent edges (4 or 6) of each edge.
extern const std::array<edge_bitset, 14> NeighborMasks;

// List of three edges which point to the vertices of each faces.
extern const std::array<edge_bitset, 24> FaceEdges;

// List of three faces which are adjacent to each faces.
extern const std::array<face_bitset, 24> NeighborFaces;

// List of pairs of edges for each edge.
// e.g. { 1, 9 } for edge 0: edge 1 and 9 are adjacent to edge 0
// and all three edges are coplanar.
extern const detail::neighbor_edge_pairs NeighborEdgePairs;

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

  static std::vector<face_bitset> get_holes_impl(face_bitset face_set) {
    std::vector<face_bitset> holes;

    if (face_set == 0) {
      // no holes
      return holes;
    }

    face_bitset remaining_faces = face_set;

    while (remaining_faces != 0) {
      // visit a new hole
      face_bitset to_visit_faces = 1 << bit_peek(remaining_faces);
      face_bitset visited_faces = 0;

      while (to_visit_faces != 0) {
        // scan to_visit_faces and construct its neighbor list
        face_bitset neighbors = 0;
        do {
          face_index face_idx = bit_peek(to_visit_faces);
          face_bitset visiting = 1 << face_idx;

          // move current face from to_visit_faces to visited_faces
          to_visit_faces ^= visiting;
          visited_faces |= visiting;

          neighbors |= NeighborFaces.at(face_idx);
        } while (to_visit_faces != 0);

        // update to_visit_faces
        to_visit_faces = neighbors & (~visited_faces & remaining_faces);
      }

      remaining_faces ^= visited_faces;
      holes.push_back(visited_faces);
    }

    return holes;
  }

  static std::vector<edge_bitset> get_surfaces_impl(edge_bitset edge_set) {
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

  // Vertex clustering decision tree
  //
  //   # of surfaces
  //   | = 0: No surface
  //   | = 1: # of holes
  //   |      | = 0: Closed surface
  //   |      | = 1: Simple surface -> cluster
  //   |      | ≥ 2: Multiple holes
  //   | ≥ 2: # of holes
  //          | = 1: Multiple surfaces -> cluster each surface
  //          | ≥ 2: Multiple surfaces and multiple holes,
  //                 e.g. 0b10'1111'0010'1011
  void cluster(std::vector<geometry::point3d>& vertices,
               std::unordered_map<vertex_index, vertex_index>& cluster_map) const {
    auto surfaces = get_surfaces();
    auto holes = get_holes();

    if (surfaces.size() >= 1 && holes.size() == 1) {
      // Simple or multiple surfaces.
      for (auto surface : surfaces) {
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
  }

  face_bitset get_faces() const {
    edge_bitset face_bits = 0;
    for (face_index fi = 0; fi < 24; fi++) {
      auto face_edges = FaceEdges.at(fi);
      auto face_bit = static_cast<int>((intersections & face_edges) == face_edges);
      face_bits |= face_bit << fi;
    }
    return face_bits;
  }

  std::vector<face_bitset> get_holes() const {
    face_bitset hole_faces = ~get_faces() & FaceSetMask;
    return get_holes_impl(hole_faces);
  }

  std::vector<edge_bitset> get_surfaces() const {
    // The most common case
    if (intersections == 0) {
      return {};
    }

    return get_surfaces_impl(intersections);
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
