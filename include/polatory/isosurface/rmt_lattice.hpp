// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <omp.h>

#include <polatory/common/bsearch.hpp>
#include <polatory/common/uncertain.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/rmt_node.hpp>
#include <polatory/isosurface/rmt_node_list.hpp>
#include <polatory/isosurface/rmt_primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>

namespace polatory {
namespace isosurface {

namespace {

constexpr std::array<edge_index, 14> OppositeEdge
  {
    7, 8, 9, 10, 11, 12, 13,
    0, 1, 2, 3, 4, 5, 6
  };

} // namespace


class rmt_lattice : public rmt_primitive_lattice {
  friend class rmt_surface;

  typedef rmt_primitive_lattice base;

  rmt_node_list node_list;
  std::vector<cell_index> nodes_to_evaluate;
  std::vector<cell_index> cells_to_visit;

  std::vector<Eigen::Vector3d> vertices;
  vertex_index clustered_vertices_begin;
  std::map<vertex_index, vertex_index> cluster_map;
  std::vector<vertex_index> unclustered_vis;

  // Add missing nodes of the eight vertices of the cell.
  // Returns false if the cell is already checked.
  bool add_cell(cell_index cell_idx) {
    auto it = node_list.find(cell_idx);
    if (it != node_list.end()) {
      if (it->second.cell_is_visited)
        return false;

      cells_to_visit.push_back(cell_idx);
    }

    bool aaa_is_added = add_node(cell_idx);
    add_node(node_list.neighbor_cell_index(cell_idx, 4));
    add_node(node_list.neighbor_cell_index(cell_idx, 9));
    add_node(node_list.neighbor_cell_index(cell_idx, 3));
    add_node(node_list.neighbor_cell_index(cell_idx, 13));
    add_node(node_list.neighbor_cell_index(cell_idx, 1));
    add_node(node_list.neighbor_cell_index(cell_idx, 12));
    add_node(node_list.neighbor_cell_index(cell_idx, 0));

    if (aaa_is_added) {
      cells_to_visit.push_back(cell_idx);
    }

    return true;
  }

  bool add_node(cell_index cell_idx) {
    if (node_list.count(cell_idx) != 0)
      return false;

    return add_node(cell_idx, cell_vector_from_index(cell_idx));
  }

  // Adds a node at cell_idx to node_list and nodes_to_evaluate
  // if the node is within the boundary.
  bool add_node(cell_index cell_idx, const cell_vector& cv) {
    Eigen::Vector3d pos = point_from_cell_vector(cv);

    if (!is_inside_bounds(pos)) {
      // Do not insert a node outside the boundary.
      return false;
    }

    auto new_node = rmt_node(pos);
    auto it_bool = node_list.insert(std::make_pair(cell_idx, std::move(new_node)));
    if (!it_bool.second) {
      // Insertion failed (the node already exists).
      assert(false);
      return false;
    }

    nodes_to_evaluate.push_back(cell_idx);
    return true;
  }

  vertex_index clustered_vertex_index(vertex_index vi) const {
    return cluster_map.count(vi) != 0
           ? cluster_map.at(vi)
           : vi;
  }

  // Evaluates field values for each node in nodes_to_evaluate.
  void evaluate_field(const field_function& field_func, double isovalue = 0.0) {
    if (nodes_to_evaluate.empty())
      return;

    std::random_device rd;
    std::minstd_rand gen(rd());
    std::uniform_real_distribution<double> dis(-1e-10, 1e-10);

    std::vector<Eigen::Vector3d> points;

    for (auto idx : nodes_to_evaluate) {
      points.push_back(node_list.at(idx).position());
    }

    auto values = field_func(points);
    values.array() -= isovalue;

    auto i = 0;
    for (auto idx : nodes_to_evaluate) {
      auto value = values(i);
      while (value == 0.0) {
        // TODO: Take the variance of the field value into account?
        value = dis(gen);
      }

      node_list.at(idx).set_value(value);
      i++;
    }

    nodes_to_evaluate.clear();
  }

  // Removes nodes without any intersections.
  void remove_free_nodes(const std::vector<cell_index>& nodes) {
    for (auto cell_idx : nodes) {
      auto it = node_list.find(cell_idx);
      if (it->second.all_intersections == 0)
        node_list.erase(it->first);
    }
  }

  // Returns the number of cells added.
  int track_surface() {
    std::set<cell_index> cells_to_add;

    // Check 12 edges of each cell and add neighbor cells adjacent to an edge
    // at which ends the field values take opposite signs.
    for (auto cell_idx : cells_to_visit) {
      const auto iaaa = cell_idx;
      const auto iaab = node_list.neighbor_cell_index(cell_idx, 4);
      const auto iaba = node_list.neighbor_cell_index(cell_idx, 9);
      const auto iabb = node_list.neighbor_cell_index(cell_idx, 3);
      const auto ibaa = node_list.neighbor_cell_index(cell_idx, 13);
      const auto ibab = node_list.neighbor_cell_index(cell_idx, 1);
      const auto ibba = node_list.neighbor_cell_index(cell_idx, 12);
      const auto ibbb = node_list.neighbor_cell_index(cell_idx, 0);

      const auto aaa = node_list.node_ptr(iaaa);
      const auto aab = node_list.node_ptr(iaab);
      const auto aba = node_list.node_ptr(iaba);
      const auto abb = node_list.node_ptr(iabb);
      const auto baa = node_list.node_ptr(ibaa);
      const auto bab = node_list.node_ptr(ibab);
      const auto bba = node_list.node_ptr(ibba);
      const auto bbb = node_list.node_ptr(ibbb);

      const auto saaa = aaa ? aaa->value_sign() : common::uncertain<binary_sign>();
      const auto saab = aab ? aab->value_sign() : common::uncertain<binary_sign>();
      const auto saba = aba ? aba->value_sign() : common::uncertain<binary_sign>();
      const auto sabb = abb ? abb->value_sign() : common::uncertain<binary_sign>();
      const auto sbaa = baa ? baa->value_sign() : common::uncertain<binary_sign>();
      const auto sbab = bab ? bab->value_sign() : common::uncertain<binary_sign>();
      const auto sbba = bba ? bba->value_sign() : common::uncertain<binary_sign>();
      const auto sbbb = bbb ? bbb->value_sign() : common::uncertain<binary_sign>();

      // __a and __b
      if (aaa && aab && saaa.get() != saab.get()) {  // o -> 4
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 5));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 6));
      }
      if (aba && abb && saba.get() != sabb.get()) {  // 9 -> 3
        cells_to_add.insert(iaba);
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 5));
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 6));
      }
      if (baa && bab && sbaa.get() != sbab.get()) {  // 13 -> 1
        cells_to_add.insert(ibaa);
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 5));
      }
      if (bba && bbb && sbba.get() != sbbb.get()) {  // 12 -> 0
        cells_to_add.insert(ibba);
        cells_to_add.insert(node_list.neighbor_cell_index(ibba, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(ibba, 6));
      }

      // _a_ and _b_
      if (aaa && aba && saaa.get() != saba.get()) {  // o -> 9
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 6));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 8));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 11));
      }
      if (aab && abb && saab.get() != sabb.get()) {  // 4 -> 3
        cells_to_add.insert(iaab);
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 6));
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 8));
      }
      if (baa && bba && sbaa.get() != sbba.get()) {  // 13 -> 12
        cells_to_add.insert(ibaa);
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 8));
        cells_to_add.insert(node_list.neighbor_cell_index(ibaa, 11));
      }
      if (bab && bbb && sbab.get() != sbbb.get()) {  // 1 -> 0
        cells_to_add.insert(ibab);
        cells_to_add.insert(node_list.neighbor_cell_index(ibab, 6));
        cells_to_add.insert(node_list.neighbor_cell_index(ibab, 11));
      }

      // a__ and b__
      if (aaa && baa && saaa.get() != sbaa.get()) {  // o -> 13
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 10));
        cells_to_add.insert(node_list.neighbor_cell_index(iaaa, 11));
      }
      if (aab && bab && saab.get() != sbab.get()) {  // 4 -> 1
        cells_to_add.insert(iaab);
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iaab, 10));
      }
      if (aba && bba && saba.get() != sbba.get()) {  // 9 -> 12
        cells_to_add.insert(iaba);
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 10));
        cells_to_add.insert(node_list.neighbor_cell_index(iaba, 11));
      }
      if (abb && bbb && sabb.get() != sbbb.get()) {  // 3 -> 0
        cells_to_add.insert(iabb);
        cells_to_add.insert(node_list.neighbor_cell_index(iabb, 2));
        cells_to_add.insert(node_list.neighbor_cell_index(iabb, 11));
      }

      aaa->cell_is_visited = true;
    }

    cells_to_visit.clear();

    int count = 0;
    for (auto cell_idx : cells_to_add) {
      if (add_cell(cell_idx))
        count++;
    }

    return count;
  }

  void update_neighbor_cache() {
    for (auto& nodei : node_list) {
      auto cell_idx = nodei.first;
      auto& node = nodei.second;

      node.neighbor_cache.reset(new rmt_node *[14]);

      for (int ei = 0; ei < 14; ei++) {
        node.neighbor_cache[ei] = node_list.neighbor_node_ptr(cell_idx, ei);
      }
    }
  }

public:
  rmt_lattice(const geometry::bbox3d& bbox, double resolution)
    : base(bbox, resolution)
    , clustered_vertices_begin(0) {
    node_list.init_strides(cell_index{ 1 } << shift1, cell_index{ 1 } << shift2);
  }

  // Add all nodes inside the boundary.
  void add_all_nodes(const field_function& field_func, double isovalue = 0.0) {
    std::vector<cell_index> new_nodes;
    std::vector<cell_index> prev_nodes;

    for (int m2 = cell_min(2); m2 <= cell_max(2); m2++) {
      cell_index offset2 = static_cast<cell_index>(m2 - cell_min(2)) << shift2;

      for (int m1 = cell_min(1); m1 <= cell_max(1); m1++) {
        cell_index offset21 = offset2 | (static_cast<cell_index>(m1 - cell_min(1)) << shift1);

        for (int m0 = cell_min(0); m0 <= cell_max(0); m0++) {
          cell_index cell_idx = offset21 | static_cast<cell_index>(m0 - cell_min(0));

          if (add_node(cell_idx, cell_vector(m0, m1, m2)))
            new_nodes.push_back(cell_idx);
        }
      }

      if (m2 > cell_min(2)) {
        evaluate_field(field_func, isovalue);
        generate_vertices(prev_nodes);
        remove_free_nodes(prev_nodes);
      }

      prev_nodes.swap(new_nodes);
      new_nodes.clear();
    }

    remove_free_nodes(prev_nodes);

    update_neighbor_cache();
  }

  void add_nodes_by_tracking(const field_function& field_func, double isovalue = 0.0) {
    evaluate_field(field_func, isovalue);
    while (track_surface() > 0) {
      evaluate_field(field_func, isovalue);
    }

    std::vector<cell_index> all_nodes;
    for (auto const& node_map : node_list)
      all_nodes.push_back(node_map.first);

    generate_vertices(all_nodes);
    remove_free_nodes(all_nodes);

    update_neighbor_cache();
  }

  // TODO: Perform gradient search to find a right cell where the isosurface passes.
  bool add_cell_contains_point(const Eigen::Vector3d& p) {
    auto cell_idx = cell_contains_point(p);
    return add_cell(cell_idx);
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

    for (auto& nodei : node_list) {
      auto& node = nodei.second;
      node.cluster(vertices, cluster_map);
    }
  }

  void generate_vertices(const std::vector<cell_index>& nodes) {
    static constexpr std::array<edge_index, 7> CellEdgeIndices{ 0, 1, 3, 4, 9, 12, 13 };

#pragma omp parallel
    {
      size_t thread_count = omp_get_num_threads();
      size_t thread_num = omp_get_thread_num();
      auto map_size = nodes.size();
      auto map_it = nodes.begin();
      if (thread_num < map_size)
        std::advance(map_it, thread_num);

      for (auto i = thread_num; i < map_size; i += thread_count) {
        auto cell_idx = *map_it;
        auto& node = node_list.at(cell_idx);

        // "distance" to the intersection point from the node
        double d = std::abs(node.value());

        for (edge_index ei : CellEdgeIndices) {
          auto it = node_list.find_neighbor_node(cell_idx, ei);
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
          double d2 = std::abs(node2.value());
          auto vertex = (d2 * node.position() + d * node2.position()) / (d + d2);

#pragma omp critical
          {
            vertex_index vi = vertices.size();
            vertices.push_back(vertex);

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

  const std::vector<Eigen::Vector3d>& get_vertices() const {
    return vertices;
  }

  // This method should be called right after calling uncluster_vertices().
  void remove_unreferenced_vertices() {
    std::vector<vertex_index> vimap(vertices.size());
    std::vector<Eigen::Vector3d> reduced_vertices;
    reduced_vertices.reserve(vertices.size() / 3);

    for (vertex_index vi = 0; vi < vertices.size(); vi++) {
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

    for (auto& nodei : node_list) {
      auto& node = nodei.second;
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

} // namespace isosurface
} // namespace polatory
