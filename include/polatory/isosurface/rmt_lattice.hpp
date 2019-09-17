// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <polatory/common/bsearch.hpp>
#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/rmt_node.hpp>
#include <polatory/isosurface/rmt_node_list.hpp>
#include <polatory/isosurface/rmt_primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace isosurface {

extern const std::array<edge_index, 14> OppositeEdge;

class rmt_lattice : public rmt_primitive_lattice {
  friend class rmt_surface;

  using base = rmt_primitive_lattice;

  rmt_node_list node_list;
  std::vector<cell_index> nodes_to_evaluate;
  std::unordered_set<cell_index> added_cells;
  std::vector<cell_index> last_added_cells;

  std::vector<geometry::point3d> vertices;
  vertex_index clustered_vertices_begin;
  std::unordered_map<vertex_index, vertex_index> cluster_map;
  std::vector<vertex_index> unclustered_vis;

  static bool has_intersection(const rmt_node *a, const rmt_node *b) {
    return a != nullptr && b != nullptr && a->value_sign() != b->value_sign();
  }

  // Add nodes corresponding to eight vertices of the cell.
  void add_cell(cell_index ci) {
    if (added_cells.count(ci) != 0) {
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
    if (node_list.count(ci) != 0) {
      return false;
    }

    return add_node_unchecked(ci, to_cell_vector(ci));
  }

  bool add_node_unchecked(cell_index ci, const cell_vector& cv) {
    auto p = cell_node_point(cv);

    if (!extended_bbox().contains(p)) {
      return false;
    }

    rmt_node node(p);
    node_list.insert(std::make_pair(ci, std::move(node)));

    nodes_to_evaluate.push_back(ci);
    return true;
  }

  vertex_index clustered_vertex_index(vertex_index vi) const {
    return cluster_map.count(vi) != 0
           ? cluster_map.at(vi)
           : vi;
  }

  // Evaluates field values for each node in nodes_to_evaluate.
  void evaluate_field(const field_function& field_fn, double isovalue) {
    if (nodes_to_evaluate.empty()) {
      return;
    }

    std::random_device rd;
    std::minstd_rand gen(rd());
    std::uniform_real_distribution<double> dis(-1e-10, 1e-10);

    geometry::points3d points(nodes_to_evaluate.size(), 3);

    auto point_it = common::row_begin(points);
    for (auto idx : nodes_to_evaluate) {
      *point_it++ = node_list.at(idx).position();
    }

    common::valuesd values = field_fn(points).array() - isovalue;

    auto i = 0;
    for (auto idx : nodes_to_evaluate) {
      auto value = values(i);
      while (value == 0.0) {
        value = dis(gen);
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

      const auto aaa = node_list.node_ptr(iaaa);
      const auto aab = node_list.node_ptr(iaab);
      const auto aba = node_list.node_ptr(iaba);
      const auto abb = node_list.node_ptr(iabb);
      const auto baa = node_list.node_ptr(ibaa);
      const auto bab = node_list.node_ptr(ibab);
      const auto bba = node_list.node_ptr(ibba);
      const auto bbb = node_list.node_ptr(ibbb);

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
      if (added_cells.count(ci) != 0) {
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
  rmt_lattice(const geometry::bbox3d& bbox, double resolution)
    : base(bbox, resolution)
    , clustered_vertices_begin(0) {
    node_list.init_strides(cell_index{ 1 } << shift1, cell_index{ 1 } << shift2);
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
    unclustered_vis.clear();
    vertices.clear();
  }

  void cluster_vertices() {
    clustered_vertices_begin = std::distance(vertices.begin(), vertices.end());

    for (auto& ci_node : node_list) {
      auto& node = ci_node.second;
      node.cluster(vertices, cluster_map);
    }
  }

  void generate_vertices(const std::vector<cell_index>& nodes) {
    static constexpr std::array<edge_index, 7> CellEdgeIndices{ 0, 1, 3, 4, 9, 12, 13 };

#pragma omp parallel
    {
      auto thread_count = static_cast<size_t>(omp_get_num_threads());
      auto thread_num = static_cast<size_t>(omp_get_thread_num());
      auto map_size = nodes.size();
      auto map_it = nodes.begin();
      if (thread_num < map_size)
        std::advance(map_it, thread_num);

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
              node2.insert_vertex(vi, OppositeEdge[ei]);
            }

            node.set_intersection(ei);
            node2.set_intersection(OppositeEdge[ei]);
          }
        }

        if (i + thread_count < map_size)
          std::advance(map_it, thread_count);
      }
    }
  }

  std::vector<geometry::point3d>& get_vertices() {
    return vertices;
  }

  // This method should be called right after calling uncluster_vertices().
  void remove_unreferenced_vertices() {
    auto n_vertices = static_cast<vertex_index>(vertices.size());
    std::vector<vertex_index> vimap(n_vertices);
    std::vector<geometry::point3d> reduced_vertices;
    reduced_vertices.reserve(n_vertices / 3);

    for (vertex_index vi = 0; vi < n_vertices; vi++) {
      if (cluster_map.count(vi) != 0) {
        // The vertex is clustered and no longer used.
        vimap[vi] = -1;
      } else {
        if (common::bsearch_eq(unclustered_vis.begin(), unclustered_vis.end(),
                               vi) != unclustered_vis.end()) {
          // The vertex has been unclustered.
          vimap[vi] = -1;
        } else {
          vimap[vi] = reduced_vertices.size();
          reduced_vertices.push_back(vertices[vi]);
        }
      }
    }

    for (auto& ci_node : node_list) {
      auto& node = ci_node.second;
      if (node.vis) {
        for (auto& vi : *node.vis) {
          vi = vimap[clustered_vertex_index(vi)];
        }
      }
    }

    vertices = reduced_vertices;

    cluster_map.clear();
    unclustered_vis.clear();
  }

  template <class InputIterator>
  void uncluster_vertices(InputIterator vis_begin, InputIterator vis_end) {
    unclustered_vis.clear();
    for (auto it = vis_begin; it != vis_end; ++it) {
      if (*it >= clustered_vertices_begin)
        unclustered_vis.push_back(*it);
    }
    std::sort(unclustered_vis.begin(), unclustered_vis.end());

    auto map_it = cluster_map.begin();
    while (map_it != cluster_map.end()) {
      if (common::bsearch_eq(unclustered_vis.begin(), unclustered_vis.end(),
                             map_it->second) != unclustered_vis.end()) {
        // Uncluster.
        map_it = cluster_map.erase(map_it);
      } else {
        ++map_it;
      }
    }
  }
};

}  // namespace isosurface
}  // namespace polatory
