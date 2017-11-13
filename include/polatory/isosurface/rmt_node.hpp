// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include <Eigen/Geometry>

#include <polatory/common/uncertain.hpp>
#include <polatory/isosurface/bit.hpp>
#include <polatory/isosurface/types.hpp>

namespace polatory {
namespace isosurface {

namespace {

// Encodes 0 or 1 on 14 outgoing halfedges for each node.
typedef unsigned short edge_bitset;
// Encodes 0 or 1 on 24 faces for each node.
typedef unsigned int face_bitset;

constexpr int FaceSetMask = 0xffffff;

// Edge index per node: 0 - 13
typedef int edge_index;
// 0 - 23
typedef int face_index;

// Adjacent edges (4 or 6) of each edge.
constexpr std::array<edge_bitset, 14> NeighborMasks
  {
    0x321a, 0x2015, 0x24b2, 0x0251, 0x006f, 0x00d4, 0x03b8,
    0x0d64, 0x0ac0, 0x1949, 0x2884, 0x3780, 0x2a01, 0x1c07
  };

// List of three edges which point to the vertices of each faces.
constexpr std::array<edge_bitset, 24> FaceEdges
  {
    0x0013, 0x0209, 0x0019, 0x1201, 0x3001, 0x2003,
    0x0016, 0x2006, 0x0034, 0x00a4, 0x0484, 0x2404,
    0x0058, 0x0248, 0x0070, 0x00e0, 0x01c0, 0x0340,
    0x0c80, 0x0980, 0x0b00, 0x1a00, 0x2c00, 0x3800
  };

// List of three faces which are adjacent to each faces.
constexpr std::array<face_bitset, 24> NeighborFaces
  {
    0x000064, 0x00200c, 0x001003, 0x200012, 0x800028, 0x000091,
    0x000181, 0x000860, 0x004240, 0x008500, 0x040a00, 0x400480,
    0x006004, 0x021002, 0x009100, 0x014200, 0x0a8000, 0x112000,
    0x480400, 0x150000, 0x2a0000, 0x900008, 0x840800, 0x600010
  };

// List of pairs of edges for each edge.
// e.g. { 1, 9 } for edge 0: edge 1 and 9 are adjacent to edge 0
// and all three edges are coplanar.
std::array<std::vector<std::pair<int, int>>, 14> NeighborEdgePairs
  {{
     {
       { 1, 9 },
       { 3, 13 },
       { 4, 12 }
     },
     {
       { 2, 0 },
       { 4, 13 }
     },
     {
       { 1, 7 },
       { 4, 10 },
       { 5, 13 }
     },
     {
       { 0, 6 },
       { 4, 9 }
     },
     {
       { 0, 5 },
       { 1, 6 },
       { 2, 3 }
     },
     {
       { 2, 6 },
       { 4, 7 }
     },
     {
       { 3, 7 },
       { 4, 8 },
       { 5, 9 }
     },
     {
       { 2, 8 },
       { 5, 11 },
       { 6, 10 }
     },
     {
       { 6, 11 },
       { 7, 9 }
     },
     {
       { 0, 8 },
       { 3, 11 },
       { 6, 12 }
     },
     {
       { 2, 11 },
       { 7, 13 }
     },
     {
       { 7, 12 },
       { 8, 13 },
       { 9, 10 }
     },
     {
       { 0, 11 },
       { 9, 13 }
     },
     {
       { 0, 10 },
       { 1, 11 },
       { 2, 12 }
     }
   }};

enum binary_sign {
  Pos = 0, Neg = 1
};

} // namespace


class rmt_node {
  Eigen::Vector3d pos;
  double val;

public:
  bool cell_is_visited;
  bool evaluated;

  // The corresponding bit is set if an edge crosses the isosurface
  // at a point nearer than the midpoint.
  // Such intersections are called "near intersections".
  edge_bitset intersections;

  // The corresponding bit is set if an edge crosses the isosurface.
  edge_bitset all_intersections;

  std::unique_ptr<std::vector<vertex_index>> vis;

  mutable std::unique_ptr<rmt_node *[]> neighbor_cache;

private:
  static std::vector<face_bitset> get_holes_impl(face_bitset face_set) {
    std::vector<face_bitset> holes;

    if (face_set == 0) {
      // no holes
      return holes;
    }

    face_bitset remaining_faces = face_set;

    while (remaining_faces != 0) {
      // visit a new hole
      face_bitset to_visit_faces = 1 << bit::peek(remaining_faces);
      face_bitset visited_faces = 0;

      while (to_visit_faces != 0) {
        // scan to_visit_faces and construct its neighbor list
        face_bitset neighbors = 0;
        do {
          face_index face_idx = bit::peek(to_visit_faces);
          face_bitset visiting = 1 << face_idx;

          // move current face from to_visit_faces to visited_faces
          to_visit_faces ^= visiting;
          visited_faces |= visiting;

          neighbors |= NeighborFaces[face_idx];
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
      edge_bitset to_visit_edges = 1 << bit::peek(remaining_edges);
      edge_bitset visited_edges = 0;

      while (to_visit_edges != 0) {
        // scan to_visit_edges and build its neighbor list
        edge_bitset neighbors = 0;
        do {
          edge_index edge_idx = bit::peek(to_visit_edges);
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
    if (edge_set == 0) return 0;

    edge_bitset neighbors = 0;
    do {
      edge_index edge_idx = bit::pop(edge_set);
      neighbors |= NeighborMasks[edge_idx];
    } while (edge_set != 0);

    return neighbors;
  }

public:
  explicit rmt_node(const Eigen::Vector3d& position)
    : pos(position)
    , val(0.0)
    , cell_is_visited(false)
    , evaluated(false)
    , intersections(0)
    , all_intersections(0)
    , vis(nullptr)
    , neighbor_cache(nullptr) {
  }

  rmt_node(rmt_node&& other) noexcept
    : pos(other.pos)
    , val(other.val)
    , cell_is_visited(other.cell_is_visited)
    , evaluated(other.evaluated)
    , intersections(other.intersections)
    , all_intersections(other.all_intersections)
    , vis(std::move(other.vis))
    , neighbor_cache(std::move(other.neighbor_cache)) {
  }

  // Vertex clustering decision tree
  //
  //   # of surfaces
  //   |  = 0 -> no surface
  //   | >= 2 -> # of holes
  //   |         |  = 1 -> Multiple surfaces
  //   |         | >= 2 -> Multiple surfaces and multiple holes,
  //   |         |         do not cluster
  //   |         |         e.g. 0b10'1111'0010'1011
  //   |         .
  //   |  = 1 -> # of holes
  //   |         |  = 0 -> Closed surface
  //   |         | >= 2 -> Multiple holes
  //   |         |  = 1 -> Simple surface
  //   |         .
  //   .
  void cluster(std::vector<Eigen::Vector3d>& vertices, std::map<vertex_index, vertex_index>& cluster_map) const {
    auto surfaces = get_surfaces();
    auto holes = get_holes();

    if (surfaces.size() == 1 && holes.size() == 1) {
      // simple surface
      auto surface = surfaces[0];

      std::vector<double> weights;

      if (bit::count(surface) == 1) {
        weights.push_back(1.0);
      } else {
        while (surface) {
          auto edge_idx = bit::pop(surface);
          auto weight = clustering_weight(edge_idx);
          if (!weight.is_certain()) return;
          weights.push_back(weight.get());
        }
      }
      auto weights_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
      assert(weights_sum > 0.0);

      Eigen::Vector3d clustered = Eigen::Vector3d::Zero();
      for (size_t i = 0; i < weights.size(); i++) {
        auto vi = (*vis)[i];
        clustered += weights[i] / weights_sum * vertices[vi];
      }

      vertex_index new_vi = vertices.size();
      vertices.push_back(clustered);

      for (auto vi : *vis) {
        cluster_map.insert({ vi, new_vi });
      }
    } else if (surfaces.size() >= 2 && holes.size() == 1) {
      // multiple surfaces
      for (auto surface : surfaces) {
        std::vector<double> weights;
        std::vector<edge_index> edge_idcs;

        if (bit::count(surface) == 1) {
          weights.push_back(1.0);
          edge_idcs.push_back(bit::pop(surface));
        } else {
          while (surface) {
            auto edge_idx = bit::pop(surface);
            auto weight = clustering_weight(edge_idx);
            if (!weight.is_certain()) return;
            weights.push_back(weight.get());
            edge_idcs.push_back(edge_idx);
          }
        }
        auto weights_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        assert(weights_sum > 0.0);

        Eigen::Vector3d clustered = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < weights.size(); i++) {
          auto edge_idx = edge_idcs[i];
          auto vi = vertex_on_edge(edge_idx);
          clustered += weights[i] / weights_sum * vertices[vi];
        }

        vertex_index new_vi = vertices.size();
        vertices.push_back(clustered);

        for (auto edge_idx : edge_idcs) {
          cluster_map.insert({ vertex_on_edge(edge_idx), new_vi });
        }
      }
    }
  }

  common::uncertain<double> clustering_weight(edge_index edge_idx) const {
    if (neighbor_cache[edge_idx] == nullptr)
      return common::uncertain<double>();

    auto& a_node = *neighbor_cache[edge_idx];
    Eigen::Vector3d oa = a_node.pos - pos;

    std::vector<double> alphas;
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();

    // Calculate alphas and accumulate normals per plane.

    for (auto neigh_pair : NeighborEdgePairs[edge_idx]) {
      if (neighbor_cache[neigh_pair.first] == nullptr ||
          neighbor_cache[neigh_pair.second] == nullptr)
        return common::uncertain<double>();

      auto& b_node = *neighbor_cache[neigh_pair.first];
      auto& c_node = *neighbor_cache[neigh_pair.second];

      auto theta_b = clustering_weight_theta(*this, a_node, b_node);
      auto theta_c = clustering_weight_theta(*this, a_node, c_node);

      alphas.push_back(std::abs(theta_b) + std::abs(theta_c));

      auto ob = b_node.pos - pos;
      auto oc = c_node.pos - pos;

      // Orthogonalize ob and oc against oa.
      auto ob_ortho = ob - ob.dot(oa) / oa.dot(oa) * oa;
      auto oc_ortho = oc - oc.dot(oa) / oa.dot(oa) * oa;

      normal += ob_ortho.normalized() / std::tan(theta_b);
      normal += oc_ortho.normalized() / std::tan(theta_c);
    }

    normal += oa.normalized();
    normal.normalize();

    // Calculate weights per plane

    std::vector<double> weights;

    for (size_t i = 0; i < alphas.size(); i++) {
      auto neigh_pair = NeighborEdgePairs[edge_idx][i];
      auto& b_node = *neighbor_cache[neigh_pair.first];

      Eigen::Vector3d ob = b_node.pos - pos;

      Eigen::Vector3d plane_normal = oa.cross(ob).normalized();

      auto cos_gamma = normal.dot(plane_normal);

      weights.push_back(std::sqrt(
        (1.0 - cos_gamma * cos_gamma) *
        (1.0 / std::pow(std::sin(alphas[i] / 2.0), 2.0) - 1.0)
      ));
    }

    return *std::max_element(weights.begin(), weights.end());
  }

  static double clustering_weight_theta(const rmt_node& o_node, const rmt_node& a_node, const rmt_node& b_node) {
    auto oa = a_node.pos - o_node.pos;
    auto dist_oa = oa.norm();

    auto d_o = o_node.value();
    auto d_a = a_node.value();
    auto d_b = b_node.value();

    if (d_b * d_a < 0) {
      // intersection on edge ob
      auto ob = b_node.pos - o_node.pos;
      auto dist_ob = ob.norm();
      auto cos_aob = oa.dot(ob) / (dist_oa * dist_ob);
      auto sin_aob = std::sqrt(1.0 - cos_aob * cos_aob);
      return std::atan(sin_aob / ((d_o - d_b) * dist_oa / ((d_o - d_a) * dist_ob) - cos_aob));
    } else {
      // intersection on edge ab
      auto ab = b_node.pos - a_node.pos;
      auto dist_ab = ab.norm();
      auto cos_oab = -oa.dot(ab) / (dist_oa * dist_ab);
      auto sin_oab = std::sqrt(1.0 - cos_oab * cos_oab);
      return std::atan(sin_oab / ((d_a - d_b) * dist_oa / ((d_a - d_o) * dist_ab) - cos_oab));
    }
  }

  face_bitset get_faces() const {
    edge_bitset face_bits = 0;
    for (face_index fi = 0; fi < 24; fi++) {
      auto face_edges = FaceEdges[fi];
      int face_bit = (intersections & face_edges) == face_edges;
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
    if (intersections == 0)
      return std::vector<edge_bitset>();

    return get_surfaces_impl(intersections);
  }

  bool has_intersection(edge_index edge_idx) const {
    edge_bitset edge_bit = 1 << edge_idx;
    return (intersections & edge_bit) != 0;
  }

  void insert_vertex(vertex_index vi, edge_index edge_idx) {
    assert(!has_intersection(edge_idx));

    if (!vis) vis.reset(new std::vector<vertex_index>);

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;

    intersections |= edge_bit;

    auto it = vis->begin() + bit::count(intersections & edge_count_mask);
    vis->insert(it, vi);

    assert(vertex_on_edge(edge_idx) == vi);
  }

  const Eigen::Vector3d& position() const {
    return pos;
  }

  void set_intersection(edge_index edge_idx) {
    edge_bitset edge_bit = 1 << edge_idx;

    all_intersections |= edge_bit;
  }

  void set_value(double value) {
    assert(!evaluated);
    this->val = value;
    evaluated = true;
  }

  double value() const {
    assert(evaluated);
    return val;
  }

  binary_sign value_sign() const {
    return value() < 0 ? Neg : Pos;
  }

  vertex_index vertex_on_edge(edge_index edge_idx) const {
    assert(has_intersection(edge_idx));

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;
    return (*vis)[bit::count(intersections & edge_count_mask)];
  }
};

} // namespace isosurface
} // namespace polatory
