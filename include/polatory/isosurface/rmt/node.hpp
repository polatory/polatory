#pragma once

#include <memory>
#include <optional>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/bit.hpp>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/sign.hpp>
#include <polatory/isosurface/types.hpp>
#include <vector>

namespace polatory::isosurface::rmt {

class Node {
  // Encodes 0 or 1 on 14 outgoing halfedges for each node.
  using EdgeBitset = std::uint16_t;

 public:
  explicit Node(const geometry::Point3& position) : position_(position) {}

  bool has_vertex(EdgeIndex edge_idx) const {
    EdgeBitset edge_bit = 1 << edge_idx;
    return (intersections_ & edge_bit) != 0;
  }

  void insert_vertex(Index vi, EdgeIndex edge_idx) {
    POLATORY_ASSERT(!has_vertex(edge_idx));

    if (!vis_) {
      vis_ = std::make_unique<std::vector<Index>>();
    }

    EdgeBitset edge_bit = 1 << edge_idx;
    EdgeBitset edge_count_mask = edge_bit - 1;

    intersections_ |= edge_bit;

    auto it = vis_->begin() + bit_count(static_cast<EdgeBitset>(intersections_ & edge_count_mask));
    vis_->insert(it, vi);

    POLATORY_ASSERT(vertex(edge_idx) == vi);
  }

  bool is_free() const { return all_intersections_ == 0; }

  const geometry::Point3& position() const { return position_; }

  void remove_vertex(EdgeIndex edge_idx) {
    POLATORY_ASSERT(has_vertex(edge_idx));

    EdgeBitset edge_bit = 1 << edge_idx;
    EdgeBitset edge_count_mask = edge_bit - 1;

    intersections_ ^= edge_bit;

    auto it = vis_->begin() + bit_count(static_cast<EdgeBitset>(intersections_ & edge_count_mask));
    vis_->erase(it);
  }

  void set_intersection(EdgeIndex edge_idx) {
    EdgeBitset edge_bit = 1 << edge_idx;

    all_intersections_ |= edge_bit;
  }

  void set_value(double value) {
    POLATORY_ASSERT(!value_.has_value());
    value_ = value;
  }

  double value() const { return value_.value(); }

  BinarySign value_sign() const { return sign(value()); }

  Index vertex(EdgeIndex edge_idx) const {
    POLATORY_ASSERT(has_vertex(edge_idx));

    EdgeBitset edge_bit = 1 << edge_idx;
    EdgeBitset edge_count_mask = edge_bit - 1;
    return vis_->at(bit_count(static_cast<EdgeBitset>(intersections_ & edge_count_mask)));
  }

 private:
  geometry::Point3 position_;
  std::optional<double> value_;

  // The corresponding bit is set if the isosurface crosses the edge
  // at a point nearer than the midpoint.
  // Such intersections are called "near intersections".
  EdgeBitset intersections_{};

  // The corresponding bit is set if the isosurface crosses the edge.
  EdgeBitset all_intersections_{};

  // Packed vertex indices for the near intersections.
  // Wrapped in a unique_ptr to reduce memory usage when the node is free.
  std::unique_ptr<std::vector<Index>> vis_;
};

}  // namespace polatory::isosurface::rmt
