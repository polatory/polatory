#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <memory>
#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/rmt_node.hpp>
#include <polatory/isosurface/rmt_node_list.hpp>
#include <polatory/isosurface/rmt_primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::isosurface {

extern const std::array<edge_index, 14> OppositeEdge;

class rmt_lattice : public rmt_primitive_lattice {
  friend class rmt_surface;

  using base = rmt_primitive_lattice;

  static constexpr double kZeroValueReplacement = 1e-10;

  rmt_node_list node_list;
  std::vector<cell_index> nodes_to_evaluate;
  std::unordered_set<cell_index> added_cells;
  std::vector<cell_index> last_added_cells;

  std::vector<geometry::point3d> vertices;
  std::unordered_map<vertex_index, vertex_index> cluster_map;

  static bool has_intersection(const rmt_node* a, const rmt_node* b) {
    return a != nullptr && b != nullptr && a->value_sign() != b->value_sign();
  }

  // Add nodes corresponding to eight vertices of the cell.
  void add_cell(cell_index ci) {
    if (added_cells.contains(ci)) {
      return;
    }

    add_node(ci);
    add_node(node_list.neighbor_cell_index(ci, 4));
    add_node(node_list.neighbor_cell_index(ci, 9));
    add_node(node_list.neighbor_cell_index(ci, 3));
    add_node(node_list.neighbor_cell_index(ci, 13));
    add_node(node_list.neighbor_cell_index(ci, 1));
    add_node(node_list.neighbor_cell_index(ci, 12));
    add_node(node_list.neighbor_cell_index(ci, 0));

    added_cells.insert(ci);
    last_added_cells.push_back(ci);
  }

  bool add_node(cell_index ci) {
    if (node_list.contains(ci)) {
      return false;
    }

    return add_node_unchecked(ci, to_cell_vector(ci));
  }

  bool add_node_unchecked(cell_index ci, const cell_vector& cv) {
    auto p = cell_node_point(cv);

    if (!extended_bbox().contains(p)) {
      return false;
    }

    node_list.emplace(ci, rmt_node{p});

    nodes_to_evaluate.push_back(ci);
    return true;
  }

  vertex_index clustered_vertex_index(vertex_index vi) const {
    return cluster_map.contains(vi) ? cluster_map.at(vi) : vi;
  }

  // Evaluates field values for each node in nodes_to_evaluate.
  void evaluate_field(const field_function& field_fn, double isovalue) {
    if (nodes_to_evaluate.empty()) {
      return;
    }

    geometry::points3d points(nodes_to_evaluate.size(), 3);

    auto point_it = common::row_begin(points);
    for (auto idx : nodes_to_evaluate) {
      *point_it++ = node_list.at(idx).position();
    }

    common::valuesd values = field_fn(points).array() - isovalue;

    auto i = 0;
    for (auto idx : nodes_to_evaluate) {
      auto value = values(i);
      if (value == 0.0) {
        value = kZeroValueReplacement;
      }

      node_list.at(idx).set_value(value);
      i++;
    }

    nodes_to_evaluate.clear();
  }

  // Removes nodes without any intersections.
  void remove_free_nodes(const std::vector<cell_index>& nodes) {
    for (auto ci : nodes) {
      auto it = node_list.find(ci);
      if (it->second.all_intersections == 0) {
        node_list.erase(it->first);
      }
    }
  }

  void track_surface() {
    std::set<cell_index> cells_to_add;

    // Check 12 edges of each cell and add neighbor cells adjacent to an edge
    // at which ends the field values take opposite signs.
    for (auto ci : last_added_cells) {
      const auto iaaa = ci;
      const auto iaab = node_list.neighbor_cell_index(ci, 4);
      const auto iaba = node_list.neighbor_cell_index(ci, 9);
      const auto iabb = node_list.neighbor_cell_index(ci, 3);
      const auto ibaa = node_list.neighbor_cell_index(ci, 13);
      const auto ibab = node_list.neighbor_cell_index(ci, 1);
      const auto ibba = node_list.neighbor_cell_index(ci, 12);
      const auto ibbb = node_list.neighbor_cell_index(ci, 0);

      const auto* aaa = node_list.node_ptr(iaaa);
      const auto* aab = node_list.node_ptr(iaab);
      const auto* aba = node_list.node_ptr(iaba);
      const auto* abb = node_list.node_ptr(iabb);
      const auto* baa = node_list.node_ptr(ibaa);
      const auto* bab = node_list.node_ptr(ibab);
      const auto* bba = node_list.node_ptr(ibba);
      const auto* bbb = node_list.node_ptr(ibbb);

      // __a and __b
      if (has_intersection(aaa, aab)) {  // o -> 4
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 5));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 6));
      }
      if (has_intersection(aba, abb)) {  // 9 -> 3
        cells_to_add.insert(iaba);
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 5));
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 6));
      }
      if (has_intersection(baa, bab)) {  // 13 -> 1
        cells_to_add.insert(ibaa);
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 5));
      }
      if (has_intersection(bba, bbb)) {  // 12 -> 0
        cells_to_add.insert(ibba);
        cells_to_add.insert(node_list.neighbor_cell_index(ibba, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(ibba, 6));
      }

      // _a_ and _b_
      if (has_intersection(aaa, aba)) {  // o -> 9
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 6));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 8));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 11));
      }
      if (has_intersection(aab, abb)) {  // 4 -> 3
        cells_to_add.insert(iaab);
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 6));
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 8));
      }
      if (has_intersection(baa, bba)) {  // 13 -> 12
        cells_to_add.insert(ibaa);
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 8));
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 11));
      }
      if (has_intersection(bab, bbb)) {  // 1 -> 0
        cells_to_add.insert(ibab);
        cells_to_add.insert(node_list.neighbor_cell_index(ibab, 6));
        cells_to_add.insert(node_list.neighbor_cell_index(ibab, 11));
      }

      // a__ and b__
      if (has_intersection(aaa, baa)) {  // o -> 13
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 10));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 11));
      }
      if (has_intersection(aab, bab)) {  // 4 -> 1
        cells_to_add.insert(iaab);
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 10));
      }
      if (has_intersection(aba, bba)) {  // 9 -> 12
        cells_to_add.insert(iaba);
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 10));
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 11));
      }
      if (has_intersection(abb, bbb)) {  // 3 -> 0
        cells_to_add.insert(iabb);
        cells_to_add.insert(node_list.neighbor_cell_index(iabb, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iabb, 11));
      }
    }

    last_added_cells.clear();

    for (auto ci : cells_to_add) {
      if (added_cells.contains(ci)) {
        continue;
      }

      add_cell(ci);
    }
  }

  void update_neighbor_cache() {
    for (auto& ci_node : node_list) {
      auto ci = ci_node.first;
      auto& node = ci_node.second;

      auto neighbors = std::make_unique<std::array<rmt_node*, 14>>();

      for (edge_index ei = 0; ei < 14; ei++) {
        neighbors->at(ei) = node_list.neighbor_node_ptr(ci, ei);
      }

      node.set_neighbors(std::move(neighbors));
    }
  }

 public:
  rmt_lattice(const geometry::bbox3d& bbox, double resolution) : base(bbox, resolution) {
    node_list.init_strides(cell_index{1} << shift1, cell_index{1} << shift2);
  }

  // Add all nodes inside the boundary.
  void add_all_nodes(const field_function& field_fn, double isovalue) {
    std::vector<cell_index> new_nodes;
    std::vector<cell_index> prev_nodes;

    for (auto cv2 = cv_min(2); cv2 <= cv_max(2); cv2++) {
      auto offset2 = static_cast<cell_index>(cv2 - cv_offset(2)) << shift2;

      for (auto cv1 = cv_min(1); cv1 <= cv_max(1); cv1++) {
        auto offset21 = offset2 | (static_cast<cell_index>(cv1 - cv_offset(1)) << shift1);

        for (auto cv0 = cv_min(0); cv0 <= cv_max(0); cv0++) {
          auto ci = offset21 | static_cast<cell_index>(cv0 - cv_offset(0));

          if (add_node_unchecked(ci, cell_vector(cv0, cv1, cv2))) {
            new_nodes.push_back(ci);
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
    while (!last_added_cells.empty()) {
      track_surface();
      evaluate_field(field_fm, isovalue);
    }

    std::vector<cell_index> all_nodes;
    for (const auto& ci_node : node_list) {
      all_nodes.push_back(ci_node.first);
    }

    generate_vertices(all_nodes);
    remove_free_nodes(all_nodes);

    update_neighbor_cache();
  }

  void add_cell_contains_point(const geometry::point3d& p) {
    if (!extended_bbox().contains(p)) {
      return;
    }

    add_cell(cell_index_from_point(p));
  }

  void clear() {
    node_list.clear();
    nodes_to_evaluate.clear();
    cluster_map.clear();
    vertices.clear();
  }

  void cluster_vertices() {
    for (auto& ci_node : node_list) {
      auto& node = ci_node.second;
      node.cluster(vertices, cluster_map);
    }
  }

  void generate_vertices(const std::vector<cell_index>& nodes) {
    static constexpr std::array<edge_index, 7> CellEdgeIndices{0, 1, 3, 4, 9, 12, 13};

#pragma omp parallel
    {
      auto thread_count = static_cast<size_t>(omp_get_num_threads());
      auto thread_num = static_cast<size_t>(omp_get_thread_num());
      auto map_size = nodes.size();
      auto map_it = nodes.begin();
      if (thread_num < map_size) {
        std::advance(map_it, thread_num);
      }

      for (auto i = thread_num; i < map_size; i += thread_count) {
        auto ci = *map_it;
        auto& node = node_list.at(ci);

        // "distance" to the intersection point from the node
        auto d = std::abs(node.value());

        for (auto ei : CellEdgeIndices) {
          auto it = node_list.find_neighbor_node(ci, ei);
          if (it == node_list.end()) {
            // There is no neighbor node on the opposite end of the edge.
            continue;
          }

          auto& node2 = it->second;
          if (node.value_sign() == node2.value_sign()) {
            // Same sign: there is no intersection on the edge.
            continue;
          }

          // "distance" to the intersection point from the neighbor node
          auto d2 = std::abs(node2.value());
          auto vertex = (d2 * node.position() + d * node2.position()) / (d + d2);

#pragma omp critical
          {
            auto vi = static_cast<vertex_index>(vertices.size());
            vertices.emplace_back(vertex);

            if (d < d2) {
              node.insert_vertex(vi, ei);
            } else {
              node2.insert_vertex(vi, OppositeEdge.at(ei));
            }

            node.set_intersection(ei);
            node2.set_intersection(OppositeEdge.at(ei));
          }
        }

        if (i + thread_count < map_size) {
          std::advance(map_it, thread_count);
        }
      }
    }
  }

  std::vector<geometry::point3d>& get_vertices() { return vertices; }

  void uncluster_vertices(const std::set<vertex_index>& vis) {
    auto it = cluster_map.begin();
    while (it != cluster_map.end()) {
      if (vis.contains(it->second)) {
        // Uncluster.
        it = cluster_map.erase(it);
      } else {
        ++it;
      }
    }
  }
};

}  // namespace polatory::isosurface
