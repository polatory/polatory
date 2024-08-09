#pragma once

#include <array>
#include <memory>
#include <optional>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/bit.hpp>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/types.hpp>
#include <unordered_map>
#include <vector>

namespace polatory::isosurface::rmt {

// Encodes 0 or 1 on 14 outgoing halfedges for each node.
using edge_bitset = std::uint16_t;

inline constexpr edge_bitset kEdgeSetMask = 0x3fff;

// Adjacent edges (4 or 6) of each edge.
inline const std::array<edge_bitset, 14> kNeighborMasks{
    0b11001000011010,  // 0
    0b10000000010101,  // 1
    0b10010010110010,  // 2
    0b00001001010001,  // 3
    0b00000001101111,  // 4
    0b00000011010100,  // 5
    0b00001110111000,  // 6
    0b00110101100100,  // 7
    0b00101011000000,  // 8
    0b01100101001001,  // 9
    0b10100010000100,  // A
    0b11011110000000,  // B
    0b10101000000001,  // C
    0b01110000000111,  // D
};

enum binary_sign { Pos = 0, Neg = 1 };

class node {
 public:
  explicit node(const geometry::point3d& position) : position_(position) {}

  void cluster(std::vector<geometry::point3d>& vertices,
               std::unordered_map<vertex_index, vertex_index>& cluster_map) const {
    auto surfaces = connected_components(intersections_);
    for (auto surface : surfaces) {
      auto holes = connected_components(surface ^ kEdgeSetMask);
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

  node& neighbor(edge_index edge) {
    POLATORY_ASSERT(has_neighbor(edge));
    return *neighbors_->at(edge);
  }

  const node& neighbor(edge_index edge) const {
    POLATORY_ASSERT(has_neighbor(edge));
    return *neighbors_->at(edge);
  }

  const geometry::point3d& position() const { return position_; }

  void remove_vertex(edge_index edge_idx) {
    POLATORY_ASSERT(has_intersection(edge_idx));

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;

    intersections_ ^= edge_bit;

    auto it = vis_->begin() + bit_count(static_cast<edge_bitset>(intersections_ & edge_count_mask));
    vis_->erase(it);
  }

  void set_intersection(edge_index edge_idx) {
    edge_bitset edge_bit = 1 << edge_idx;

    all_intersections_ |= edge_bit;
  }

  void set_neighbors(std::unique_ptr<std::array<node*, 14>> neighbors) {
    neighbors_.swap(neighbors);
  }

  void set_value(double value) {
    POLATORY_ASSERT(!value_.has_value());
    value_ = value;
  }

  double value() const { return value_.value(); }

  binary_sign value_sign() const { return value() < 0.0 ? Neg : Pos; }

  vertex_index vertex_on_edge(edge_index edge_idx) const {
    POLATORY_ASSERT(has_intersection(edge_idx));

    edge_bitset edge_bit = 1 << edge_idx;
    edge_bitset edge_count_mask = edge_bit - 1;
    return vis_->at(bit_count(static_cast<edge_bitset>(intersections_ & edge_count_mask)));
  }

 private:
  static std::vector<edge_bitset> connected_components(edge_bitset edge_set) {
    std::vector<edge_bitset> components;
    while (edge_set != 0) {
      edge_bitset component{};
      edge_bitset queue = 1 << bit_peek(edge_set);
      while (queue != 0) {
        auto e = bit_pop(&queue);
        component |= 1 << e;
        edge_set ^= 1 << e;
        queue |= kNeighborMasks.at(e) & edge_set;
      }
      components.push_back(component);
    }
    return components;
  }

  geometry::point3d position_;
  std::optional<double> value_;

  // The corresponding bit is set if an edge crosses the isosurface
  // at a point nearer than the midpoint.
  // Such intersections are called "near intersections".
  edge_bitset intersections_{};

  // The corresponding bit is set if an edge crosses the isosurface.
  edge_bitset all_intersections_{};

  std::unique_ptr<std::vector<vertex_index>> vis_;

  std::unique_ptr<std::array<node*, 14>> neighbors_;
};

}  // namespace polatory::isosurface::rmt
