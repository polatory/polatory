#pragma once

#include <array>
#include <memory>
#include <optional>
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
inline const std::array<edge_bitset, 14> NeighborMasks{0x321a, 0x2015, 0x24b2, 0x0251, 0x006f,
                                                       0x00d4, 0x03b8, 0x0d64, 0x0ac0, 0x1949,
                                                       0x2884, 0x3780, 0x2a01, 0x1c07};

enum binary_sign { Pos = 0, Neg = 1 };

class rmt_node {
  geometry::point3d position_;
  std::optional<double> value_;

  // The corresponding bit is set if an edge crosses the isosurface
  // at a point nearer than the midpoint.
  // Such intersections are called "near intersections".
  edge_bitset intersections_{};

  // The corresponding bit is set if an edge crosses the isosurface.
  edge_bitset all_intersections_{};

  std::unique_ptr<std::vector<vertex_index>> vis_;

  std::unique_ptr<std::array<rmt_node*, 14>> neighbors_;

  static std::vector<edge_bitset> connected_components(edge_bitset edge_set) {
    std::vector<edge_bitset> components;
    while (edge_set != 0) {
      edge_bitset component{};
      edge_bitset queue = 1 << bit_peek(edge_set);
      while (queue != 0) {
        auto e = bit_pop(&queue);
        component |= 1 << e;
        edge_set ^= 1 << e;
        queue |= NeighborMasks.at(e) & edge_set;
      }
      components.push_back(component);
    }
    return components;
  }

 public:
  explicit rmt_node(const geometry::point3d& position) : position_(position) {}

  void cluster(std::vector<geometry::point3d>& vertices,
               std::unordered_map<vertex_index, vertex_index>& cluster_map) const {
    auto surfaces = connected_components(intersections_);
    for (auto surface : surfaces) {
      auto holes = connected_components(surface ^ EdgeSetMask);
      if (holes.size() != 1) {
        // holes.size() == 0 -> closed surface
        // holes.size() >= 2 -> holes in the surface
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
    return (intersections_ & edge_bit) != 0;
  }

  bool has_neighbor(edge_index edge) const { return neighbors_->at(edge) != nullptr; }

  void insert_vertex(vertex_index vi, edge_index edge_idx) {
    POLATORY_ASSERT(!has_intersection(edge_idx));

    if (!vis_) {
      vis_ = std::make_unique<std::vector<vertex_index>>();
    }

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;

    intersections_ |= edge_bit;

    auto it = vis_->begin() + bit_count(static_cast<edge_bitset>(intersections_ & edge_count_mask));
    vis_->insert(it, vi);

    POLATORY_ASSERT(vertex_on_edge(edge_idx) == vi);
  }

  bool is_free() const { return all_intersections_ == 0; }

  rmt_node& neighbor(edge_index edge) {
    POLATORY_ASSERT(has_neighbor(edge));
    return *neighbors_->at(edge);
  }

  const rmt_node& neighbor(edge_index edge) const {
    POLATORY_ASSERT(has_neighbor(edge));
    return *neighbors_->at(edge);
  }

  const geometry::point3d& position() const { return position_; }

  void set_intersection(edge_index edge_idx) {
    edge_bitset edge_bit = 1 << edge_idx;

    all_intersections_ |= edge_bit;
  }

  void set_neighbors(std::unique_ptr<std::array<rmt_node*, 14>> neighbors) {
    neighbors_.swap(neighbors);
  }

  void set_value(double value) {
    POLATORY_ASSERT(!value_.has_value());
    value_ = value;
  }

  double value() const {
    POLATORY_ASSERT(value_.has_value());
    return *value_;
  }

  binary_sign value_sign() const { return value() < 0 ? Neg : Pos; }

  vertex_index vertex_on_edge(edge_index edge_idx) const {
    POLATORY_ASSERT(has_intersection(edge_idx));

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;
    return vis_->at(bit_count(static_cast<edge_bitset>(intersections_ & edge_count_mask)));
  }
};

}  // namespace polatory::isosurface
