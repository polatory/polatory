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
using edge_bitset = uint16_t;
// Encodes 0 or 1 on 24 faces for each node.
using face_bitset = uint32_t;

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
  void cluster(std::vector<geometry::point3d>& vertices,
               std::unordered_map<vertex_index, vertex_index>& cluster_map) const {
    auto surfaces = get_surfaces();
    auto holes = get_holes();

    if (surfaces.size() == 1 && holes.size() == 1) {
      // simple surface
      auto surface = surfaces[0];

      std::vector<double> weights;

      if (bit_count(surface) == 1) {
        weights.push_back(1.0);
      } else {
        while (surface != 0) {
          auto edge_idx = bit_pop(&surface);
          auto weight = clustering_weight(edge_idx);
          if (!weight.has_value()) {
            return;
          }
          weights.push_back(weight.value());
        }
      }
      auto weights_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
      POLATORY_ASSERT(weights_sum > 0.0);

      geometry::point3d clustered = geometry::point3d::Zero();
      for (size_t i = 0; i < weights.size(); i++) {
        auto vi = vis->at(i);
        clustered += weights.at(i) / weights_sum * vertices.at(vi);
      }

      auto new_vi = static_cast<vertex_index>(vertices.size());
      vertices.push_back(clustered);

      for (auto vi : *vis) {
        cluster_map.emplace(vi, new_vi);
      }
    } else if (surfaces.size() >= 2 && holes.size() == 1) {
      // multiple surfaces
      for (auto surface : surfaces) {
        std::vector<double> weights;
        std::vector<edge_index> edge_idcs;

        if (bit_count(surface) == 1) {
          weights.push_back(1.0);
          edge_idcs.push_back(bit_pop(&surface));
        } else {
          while (surface != 0) {
            auto edge_idx = bit_pop(&surface);
            auto weight = clustering_weight(edge_idx);
            if (!weight.has_value()) {
              return;
            }
            weights.push_back(weight.value());
            edge_idcs.push_back(edge_idx);
          }
        }
        auto weights_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        POLATORY_ASSERT(weights_sum > 0.0);

        geometry::point3d clustered = geometry::point3d::Zero();
        for (size_t i = 0; i < weights.size(); i++) {
          auto edge_idx = edge_idcs.at(i);
          auto vi = vertex_on_edge(edge_idx);
          clustered += weights.at(i) / weights_sum * vertices.at(vi);
        }

        auto new_vi = static_cast<vertex_index>(vertices.size());
        vertices.push_back(clustered);

        for (auto edge_idx : edge_idcs) {
          cluster_map.emplace(vertex_on_edge(edge_idx), new_vi);
        }
      }
    }
  }

  std::optional<double> clustering_weight(edge_index edge_idx) const {
    if (!has_neighbor(edge_idx)) {
      return {};
    }

    const auto& a_node = neighbor(edge_idx);
    geometry::vector3d oa = a_node.pos - pos;

    std::vector<double> alphas;
    geometry::vector3d normal = geometry::vector3d::Zero();

    // Calculate alphas and accumulate normals per plane.

    for (auto neigh_pair : NeighborEdgePairs.at(edge_idx)) {
      if (!has_neighbor(neigh_pair.first) || !has_neighbor(neigh_pair.second)) {
        return {};
      }

      const auto& b_node = neighbor(neigh_pair.first);
      const auto& c_node = neighbor(neigh_pair.second);

      auto theta_b = clustering_weight_theta(*this, a_node, b_node);
      auto theta_c = clustering_weight_theta(*this, a_node, c_node);

      alphas.push_back(std::abs(theta_b) + std::abs(theta_c));

      geometry::vector3d ob = b_node.pos - pos;
      geometry::vector3d oc = c_node.pos - pos;

      // Orthogonalize ob and oc against oa.
      geometry::vector3d ob_ortho = ob - ob.dot(oa) / oa.dot(oa) * oa;
      geometry::vector3d oc_ortho = oc - oc.dot(oa) / oa.dot(oa) * oa;

      normal += ob_ortho.normalized() / std::tan(theta_b);
      normal += oc_ortho.normalized() / std::tan(theta_c);
    }

    normal += oa.normalized();
    normal.normalize();

    // Calculate weights per plane

    std::vector<double> weights;

    for (size_t i = 0; i < alphas.size(); i++) {
      auto neigh_pair = NeighborEdgePairs.at(edge_idx).at(i);
      const auto& b_node = neighbor(neigh_pair.first);

      geometry::vector3d ob = b_node.pos - pos;

      geometry::vector3d plane_normal = oa.cross(ob).normalized();

      auto cos_gamma = normal.dot(plane_normal);

      weights.push_back(std::sqrt((1.0 - cos_gamma * cos_gamma) *
                                  (1.0 / std::pow(std::sin(alphas.at(i) / 2.0), 2.0) - 1.0)));
    }

    return *std::max_element(weights.begin(), weights.end());
  }

  static double clustering_weight_theta(const rmt_node& o_node, const rmt_node& a_node,
                                        const rmt_node& b_node) {
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
    }
    // else
    {
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
