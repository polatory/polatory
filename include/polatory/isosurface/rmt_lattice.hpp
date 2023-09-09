#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/rmt_node.hpp>
#include <polatory/isosurface/rmt_node_list.hpp>
#include <polatory/isosurface/rmt_primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::isosurface {

inline const std::array<edge_index, 14> OppositeEdge{7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6};

class rmt_lattice : public rmt_primitive_lattice {
  friend class rmt_surface;

  using base = rmt_primitive_lattice;

  static constexpr double kZeroValueReplacement = 1e-10;

  rmt_node_list node_list_;
  std::vector<cell_vector> nodes_to_evaluate_;
  std::unordered_set<cell_vector> added_cells_;
  std::vector<cell_vector> last_added_cells_;

  std::vector<geometry::point3d> vertices_;
  std::unordered_map<vertex_index, vertex_index> cluster_map_;

  static bool has_intersection(const rmt_node* a, const rmt_node* b) {
    return a != nullptr && b != nullptr && a->value_sign() != b->value_sign();
  }

  // Add nodes corresponding to eight vertices of the cell.
  void add_cell(const cell_vector& cv) {
    if (added_cells_.contains(cv)) {
      return;
    }

    add_node(cv);
    add_node(cv + NeighborCellVectors[4]);
    add_node(cv + NeighborCellVectors[9]);
    add_node(cv + NeighborCellVectors[3]);
    add_node(cv + NeighborCellVectors[13]);
    add_node(cv + NeighborCellVectors[1]);
    add_node(cv + NeighborCellVectors[12]);
    add_node(cv + NeighborCellVectors[0]);

    added_cells_.insert(cv);
    last_added_cells_.push_back(cv);
  }

  bool add_node(const cell_vector& cv) {
    if (node_list_.contains(cv)) {
      return false;
    }

    return add_node_unchecked(cv);
  }

  bool add_node_unchecked(const cell_vector& cv) {
    auto p = cell_node_point(cv);

    // Due to the numerical error in the rotation of the lattice,
    // nodes are not perfectly aligned with planes.
    // Round the position of the node to prevent creation of near-degenerate tetrahedra.
    auto unit = resolution() / 100.0;
    p = unit * (p.array() / unit).round();

    if (!extended_bbox().contains(p)) {
      return false;
    }

    node_list_.emplace(cv, rmt_node{clamp_to_bbox(p)});

    nodes_to_evaluate_.push_back(cv);
    return true;
  }

  geometry::point3d clamp_to_bbox(const geometry::point3d& p) const {
    return p.array().max(bbox().min().array()).min(bbox().max().array());
  }

  vertex_index clustered_vertex_index(vertex_index vi) const {
    return cluster_map_.contains(vi) ? cluster_map_.at(vi) : vi;
  }

  // Evaluates field values for each node in nodes_to_evaluate_.
  void evaluate_field(const field_function& field_fn, double isovalue) {
    if (nodes_to_evaluate_.empty()) {
      return;
    }

    geometry::points3d points(nodes_to_evaluate_.size(), 3);

    auto point_it = points.rowwise().begin();
    for (const auto& cv : nodes_to_evaluate_) {
      *point_it++ = node_list_.at(cv).position();
    }

    common::valuesd values = field_fn(points).array() - isovalue;

    index_t i{};
    for (const auto& cv : nodes_to_evaluate_) {
      auto value = values(i);
      if (value == 0.0) {
        value = kZeroValueReplacement;
      }

      node_list_.at(cv).set_value(value);
      i++;
    }

    nodes_to_evaluate_.clear();
  }

  // Removes nodes without any intersections.
  void remove_free_nodes(const std::vector<cell_vector>& node_cvs) {
    for (const auto& cv : node_cvs) {
      auto it = node_list_.find(cv);
      if (it->second.is_free()) {
        node_list_.erase(it->first);
      }
    }
  }

  void track_surface() {
    std::unordered_set<cell_vector> cells_to_add;

    // Check 12 edges of each cell and add neighbor cells adjacent to an edge
    // at which ends the field values take opposite signs.
    for (const auto& cv : last_added_cells_) {
      auto iaaa = cv;
      auto iaab = cv + NeighborCellVectors[4];
      auto iaba = cv + NeighborCellVectors[9];
      auto iabb = cv + NeighborCellVectors[3];
      auto ibaa = cv + NeighborCellVectors[13];
      auto ibab = cv + NeighborCellVectors[1];
      auto ibba = cv + NeighborCellVectors[12];
      auto ibbb = cv + NeighborCellVectors[0];

      const auto* aaa = node_list_.node_ptr(iaaa);
      const auto* aab = node_list_.node_ptr(iaab);
      const auto* aba = node_list_.node_ptr(iaba);
      const auto* abb = node_list_.node_ptr(iabb);
      const auto* baa = node_list_.node_ptr(ibaa);
      const auto* bab = node_list_.node_ptr(ibab);
      const auto* bba = node_list_.node_ptr(ibba);
      const auto* bbb = node_list_.node_ptr(ibbb);

      // __a and __b
      if (has_intersection(aaa, aab)) {  // o -> 4
        cells_to_add.insert(iaaa + NeighborCellVectors[2]);
        cells_to_add.insert(iaaa + NeighborCellVectors[5]);
        cells_to_add.insert(iaaa + NeighborCellVectors[6]);
      }
      if (has_intersection(aba, abb)) {  // 9 -> 3
        cells_to_add.insert(iaba);
        cells_to_add.insert(iaba + NeighborCellVectors[5]);
        cells_to_add.insert(iaba + NeighborCellVectors[6]);
      }
      if (has_intersection(baa, bab)) {  // 13 -> 1
        cells_to_add.insert(ibaa);
        cells_to_add.insert(ibaa + NeighborCellVectors[2]);
        cells_to_add.insert(ibaa + NeighborCellVectors[5]);
      }
      if (has_intersection(bba, bbb)) {  // 12 -> 0
        cells_to_add.insert(ibba);
        cells_to_add.insert(ibba + NeighborCellVectors[2]);
        cells_to_add.insert(ibba + NeighborCellVectors[6]);
      }

      // _a_ and _b_
      if (has_intersection(aaa, aba)) {  // o -> 9
        cells_to_add.insert(iaaa + NeighborCellVectors[6]);
        cells_to_add.insert(iaaa + NeighborCellVectors[8]);
        cells_to_add.insert(iaaa + NeighborCellVectors[11]);
      }
      if (has_intersection(aab, abb)) {  // 4 -> 3
        cells_to_add.insert(iaab);
        cells_to_add.insert(iaab + NeighborCellVectors[6]);
        cells_to_add.insert(iaab + NeighborCellVectors[8]);
      }
      if (has_intersection(baa, bba)) {  // 13 -> 12
        cells_to_add.insert(ibaa);
        cells_to_add.insert(ibaa + NeighborCellVectors[8]);
        cells_to_add.insert(ibaa + NeighborCellVectors[11]);
      }
      if (has_intersection(bab, bbb)) {  // 1 -> 0
        cells_to_add.insert(ibab);
        cells_to_add.insert(ibab + NeighborCellVectors[6]);
        cells_to_add.insert(ibab + NeighborCellVectors[11]);
      }

      // a__ and b__
      if (has_intersection(aaa, baa)) {  // o -> 13
        cells_to_add.insert(iaaa + NeighborCellVectors[2]);
        cells_to_add.insert(iaaa + NeighborCellVectors[10]);
        cells_to_add.insert(iaaa + NeighborCellVectors[11]);
      }
      if (has_intersection(aab, bab)) {  // 4 -> 1
        cells_to_add.insert(iaab);
        cells_to_add.insert(iaab + NeighborCellVectors[2]);
        cells_to_add.insert(iaab + NeighborCellVectors[10]);
      }
      if (has_intersection(aba, bba)) {  // 9 -> 12
        cells_to_add.insert(iaba);
        cells_to_add.insert(iaba + NeighborCellVectors[10]);
        cells_to_add.insert(iaba + NeighborCellVectors[11]);
      }
      if (has_intersection(abb, bbb)) {  // 3 -> 0
        cells_to_add.insert(iabb);
        cells_to_add.insert(iabb + NeighborCellVectors[2]);
        cells_to_add.insert(iabb + NeighborCellVectors[11]);
      }
    }

    last_added_cells_.clear();

    for (const auto& cv : cells_to_add) {
      add_cell(cv);
    }
  }

  void update_neighbor_cache() {
    for (auto& cv_node : node_list_) {
      const auto& cv = cv_node.first;
      auto& node = cv_node.second;

      auto neighbors = std::make_unique<std::array<rmt_node*, 14>>();

      for (edge_index ei = 0; ei < 14; ei++) {
        neighbors->at(ei) = node_list_.neighbor_node_ptr(cv, ei);
      }

      node.set_neighbors(std::move(neighbors));
    }
  }

 public:
  rmt_lattice(const geometry::bbox3d& bbox, double resolution) : base(bbox, resolution) {}

  // Add all nodes inside the boundary.
  void add_all_nodes(const field_function& field_fn, double isovalue) {
    std::vector<cell_vector> new_nodes;
    std::vector<cell_vector> prev_nodes;

    for (auto cv2 = cv_min(2); cv2 <= cv_max(2); cv2++) {
      for (auto cv1 = cv_min(1); cv1 <= cv_max(1); cv1++) {
        for (auto cv0 = cv_min(0); cv0 <= cv_max(0); cv0++) {
          cell_vector cv(cv0, cv1, cv2);
          if (add_node_unchecked(cv)) {
            new_nodes.push_back(cv);
          }
        }
      }

      if (cv2 > cv_min(2)) {
        evaluate_field(field_fn, isovalue);
        generate_vertices(prev_nodes);
        remove_free_nodes(prev_nodes);
      }

      prev_nodes.swap(new_nodes);
      new_nodes.clear();
    }

    remove_free_nodes(prev_nodes);

    update_neighbor_cache();
  }

  void add_nodes_by_tracking(const field_function& field_fm, double isovalue) {
    evaluate_field(field_fm, isovalue);
    while (!last_added_cells_.empty()) {
      track_surface();
      evaluate_field(field_fm, isovalue);
    }

    std::vector<cell_vector> all_nodes;
    for (const auto& cv_node : node_list_) {
      all_nodes.push_back(cv_node.first);
    }

    generate_vertices(all_nodes);
    remove_free_nodes(all_nodes);

    update_neighbor_cache();
  }

  void add_cell_contains_point(const geometry::point3d& p) {
    if (!extended_bbox().contains(p)) {
      return;
    }

    add_cell(cell_vector_from_point(p));
  }

  void clear() {
    node_list_.clear();
    nodes_to_evaluate_.clear();
    cluster_map_.clear();
    vertices_.clear();
  }

  void cluster_vertices() {
    for (auto& ci_node : node_list_) {
      auto& node = ci_node.second;
      const auto& p = node.position();
      if ((p.array() == bbox().min().array() || p.array() == bbox().max().array()).any()) {
        // Do not cluster boundary nodes' vertices.
        continue;
      }
      node.cluster(vertices_, cluster_map_);
    }
  }

  void generate_vertices(const std::vector<cell_vector>& node_cvs) {
    static constexpr std::array<edge_index, 7> CellEdgeIndices{0, 1, 3, 4, 9, 12, 13};

#pragma omp parallel for
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(node_cvs.size()); i++) {
      const auto& cv = node_cvs.at(i);
      auto& node = node_list_.at(cv);

      // "distance" to the intersection point (if exists) from the node
      auto d = std::abs(node.value());
      const auto& p = node.position();

      for (auto ei : CellEdgeIndices) {
        auto* node2_ptr = node_list_.neighbor_node_ptr(cv, ei);
        if (node2_ptr == nullptr) {
          // There is no neighbor node on the opposite end of the edge.
          continue;
        }

        auto& node2 = *node2_ptr;
        if (node.value_sign() == node2.value_sign()) {
          // There is no intersection on the edge.
          continue;
        }

        // "distance" to the intersection point from the neighbor node
        auto d2 = std::abs(node2.value());
        const auto& p2 = node2.position();

        // Do not interpolate when coordinates are the same
        // to prevent boundary vertices from being moved.
        geometry::point3d vertex =
            (p.array() == p2.array()).select(p, (d2 * p + d * p2) / (d + d2));

#pragma omp critical
        {
          auto vi = static_cast<vertex_index>(vertices_.size());
          vertices_.emplace_back(vertex);

          if (d < d2) {
            node.insert_vertex(vi, ei);
          } else {
            node2.insert_vertex(vi, OppositeEdge.at(ei));
          }

          node.set_intersection(ei);
          node2.set_intersection(OppositeEdge.at(ei));
        }
      }
    }
  }

  geometry::points3d get_vertices() {
    geometry::points3d vertices(static_cast<index_t>(vertices_.size()), 3);
    auto it = vertices.rowwise().begin();
    for (const auto& v : vertices_) {
      *it++ = v;
    }
    return vertices;
  }

  void uncluster_vertices(const std::unordered_set<vertex_index>& vis) {
    auto it = cluster_map_.begin();
    while (it != cluster_map_.end()) {
      if (vis.contains(it->second)) {
        // Uncluster.
        it = cluster_map_.erase(it);
      } else {
        ++it;
      }
    }
  }
};

}  // namespace polatory::isosurface
