#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/rmt_edge.hpp>
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

inline const std::array<edge_index, 14> OppositeEdge{
    rmt_edge::k7, rmt_edge::k8, rmt_edge::k9, rmt_edge::kA, rmt_edge::kB,
    rmt_edge::kC, rmt_edge::kD, rmt_edge::k0, rmt_edge::k1, rmt_edge::k2,
    rmt_edge::k3, rmt_edge::k4, rmt_edge::k5, rmt_edge::k6};

class rmt_lattice : public rmt_primitive_lattice {
  friend class rmt_surface;

  using base = rmt_primitive_lattice;

  static constexpr double kZeroValueReplacement = 1e-100;

  rmt_node_list node_list_;
  std::vector<cell_vector> nodes_to_evaluate_;
  std::unordered_set<cell_vector> added_cells_;
  std::vector<cell_vector> last_added_cells_;
  double value_at_arbitrary_point_{kZeroValueReplacement};

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
    add_node(cv + NeighborCellVectors[rmt_edge::k0]);
    add_node(cv + NeighborCellVectors[rmt_edge::k1]);
    add_node(cv + NeighborCellVectors[rmt_edge::k2]);
    add_node(cv + NeighborCellVectors[rmt_edge::k3]);
    add_node(cv + NeighborCellVectors[rmt_edge::k4]);
    add_node(cv + NeighborCellVectors[rmt_edge::k5]);
    add_node(cv + NeighborCellVectors[rmt_edge::k6]);

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
    value_at_arbitrary_point_ = values(0);

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
      auto iaab = cv + NeighborCellVectors[rmt_edge::k2];
      auto iaba = cv + NeighborCellVectors[rmt_edge::k0];
      auto iabb = cv + NeighborCellVectors[rmt_edge::k1];
      auto ibaa = cv + NeighborCellVectors[rmt_edge::k6];
      auto ibab = cv + NeighborCellVectors[rmt_edge::k5];
      auto ibba = cv + NeighborCellVectors[rmt_edge::k3];
      auto ibbb = cv + NeighborCellVectors[rmt_edge::k4];

      const auto* aaa = node_list_.node_ptr(iaaa);
      const auto* aab = node_list_.node_ptr(iaab);
      const auto* aba = node_list_.node_ptr(iaba);
      const auto* abb = node_list_.node_ptr(iabb);
      const auto* baa = node_list_.node_ptr(ibaa);
      const auto* bab = node_list_.node_ptr(ibab);
      const auto* bba = node_list_.node_ptr(ibba);
      const auto* bbb = node_list_.node_ptr(ibbb);

      // __a and __b
      if (has_intersection(aaa, aab)) {  // o -> 2
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::k7]);
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::kA]);
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::kD]);
      }
      if (has_intersection(aba, abb)) {  // 0 -> 1
        cells_to_add.insert(iaba);
        cells_to_add.insert(iaba + NeighborCellVectors[rmt_edge::kA]);
        cells_to_add.insert(iaba + NeighborCellVectors[rmt_edge::kD]);
      }
      if (has_intersection(baa, bab)) {  // 6 -> 5
        cells_to_add.insert(ibaa);
        cells_to_add.insert(ibaa + NeighborCellVectors[rmt_edge::k7]);
        cells_to_add.insert(ibaa + NeighborCellVectors[rmt_edge::kA]);
      }
      if (has_intersection(bba, bbb)) {  // 3 -> 4
        cells_to_add.insert(ibba);
        cells_to_add.insert(ibba + NeighborCellVectors[rmt_edge::k7]);
        cells_to_add.insert(ibba + NeighborCellVectors[rmt_edge::kD]);
      }

      // _a_ and _b_
      if (has_intersection(aaa, aba)) {  // o -> 0
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::kD]);
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::kC]);
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::k9]);
      }
      if (has_intersection(aab, abb)) {  // 2 -> 1
        cells_to_add.insert(iaab);
        cells_to_add.insert(iaab + NeighborCellVectors[rmt_edge::kD]);
        cells_to_add.insert(iaab + NeighborCellVectors[rmt_edge::kC]);
      }
      if (has_intersection(baa, bba)) {  // 6 -> 3
        cells_to_add.insert(ibaa);
        cells_to_add.insert(ibaa + NeighborCellVectors[rmt_edge::kC]);
        cells_to_add.insert(ibaa + NeighborCellVectors[rmt_edge::k9]);
      }
      if (has_intersection(bab, bbb)) {  // 5 -> 4
        cells_to_add.insert(ibab);
        cells_to_add.insert(ibab + NeighborCellVectors[rmt_edge::kD]);
        cells_to_add.insert(ibab + NeighborCellVectors[rmt_edge::k9]);
      }

      // a__ and b__
      if (has_intersection(aaa, baa)) {  // o -> 6
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::k7]);
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::k8]);
        cells_to_add.insert(iaaa + NeighborCellVectors[rmt_edge::k9]);
      }
      if (has_intersection(aab, bab)) {  // 2 -> 5
        cells_to_add.insert(iaab);
        cells_to_add.insert(iaab + NeighborCellVectors[rmt_edge::k7]);
        cells_to_add.insert(iaab + NeighborCellVectors[rmt_edge::k8]);
      }
      if (has_intersection(aba, bba)) {  // 0 -> 3
        cells_to_add.insert(iaba);
        cells_to_add.insert(iaba + NeighborCellVectors[rmt_edge::k8]);
        cells_to_add.insert(iaba + NeighborCellVectors[rmt_edge::k9]);
      }
      if (has_intersection(abb, bbb)) {  // 1 -> 4
        cells_to_add.insert(iabb);
        cells_to_add.insert(iabb + NeighborCellVectors[rmt_edge::k7]);
        cells_to_add.insert(iabb + NeighborCellVectors[rmt_edge::k9]);
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
    for (auto& cv_node : node_list_) {
      auto& node = cv_node.second;
      const auto& p = node.position();
      if ((p.array() == bbox().min().array() || p.array() == bbox().max().array()).any()) {
        // Do not cluster boundary nodes' vertices.
        continue;
      }
      node.cluster(vertices_, cluster_map_);
    }
  }

  void generate_vertices(const std::vector<cell_vector>& node_cvs) {
    static constexpr std::array<edge_index, 7> CellEdgeIndices{
        rmt_edge::k0, rmt_edge::k1, rmt_edge::k2, rmt_edge::k3,
        rmt_edge::k4, rmt_edge::k5, rmt_edge::k6};

#pragma omp parallel for
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(node_cvs.size()); i++) {
      const auto& cv = node_cvs.at(i);
      auto& node0 = node_list_.at(cv);
      const auto& p0 = node0.position();
      auto v0 = node0.value();

      for (auto ei : CellEdgeIndices) {
        auto* node1_ptr = node_list_.neighbor_node_ptr(cv, ei);
        if (node1_ptr == nullptr) {
          // There is no neighbor node on the opposite end of the edge.
          continue;
        }

        auto& node1 = *node1_ptr;
        const auto& p1 = node1.position();
        auto v1 = node1.value();

        if (v0 * v1 > 0.0) {
          // There is no intersection on the edge.
          continue;
        }

        auto t = v0 / (v0 - v1);
        geometry::point3d vertex = p0 + t * (p1 - p0);

#pragma omp critical
        {
          auto vi = static_cast<vertex_index>(vertices_.size());
          vertices_.emplace_back(vertex);

          if (t < 0.5) {
            node0.insert_vertex(vi, ei);
          } else {
            node1.insert_vertex(vi, OppositeEdge.at(ei));
          }

          node0.set_intersection(ei);
          node1.set_intersection(OppositeEdge.at(ei));
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

  void refine_vertices(const field_function& field_fn, double isovalue) {
    geometry::points3d vertices = get_vertices();
    common::valuesd vertex_values = field_fn(vertices).array() - isovalue;
    std::vector<bool> processed(vertices_.size(), false);

    for (auto& cv_node : node_list_) {
      auto& node0 = cv_node.second;
      if (node0.is_free()) {
        continue;
      }

      const auto& p0 = node0.position();
      auto v0 = node0.value();

      for (edge_index ei = 0; ei < 14; ei++) {
        if (!node0.has_intersection(ei)) {
          continue;
        }

        auto vi = node0.vertex_on_edge(ei);
        if (processed.at(vi)) {
          continue;
        }
        processed.at(vi) = true;

        auto& node1 = node0.neighbor(ei);
        const auto& p1 = node1.position();
        auto v1 = node1.value();

        auto t = v0 / (v0 - v1);
        auto vt = vertex_values(vi);
        if (vt == 0.0) {
          continue;
        }

        // Solve y = a x^2 + b x + c for a, b, c with (x, y) = (0, v0), (t, vt), (1, v1).
        auto a = ((v1 - v0) * t + v0 - vt) / (t * (1.0 - t));
        auto b = -((v1 - v0) * t * t + v0 - vt) / (t * (1.0 - t));
        auto c = v0;

        // Solve a x^2 + b x + c = 0 for x, where 0 < x < 1.
        auto [s0, s1] = solve_quadratic(a, b, c);
        auto s = 0.0 < s0 && s0 < 1.0 ? s0 : s1;

        geometry::point3d vertex = p0 + s * (p1 - p0);
        vertices_.at(vi) = vertex;

        if (s >= 0.5) {
          node0.remove_vertex(ei);
          node1.insert_vertex(vi, OppositeEdge.at(ei));
        }
      }
    }
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

  double value_at_arbitrary_point() const { return value_at_arbitrary_point_; }

 private:
  static std::pair<double, double> solve_quadratic(double a, double b, double c) {
    auto d = b * b - 4.0 * a * c;
    auto sqrt_d = std::sqrt(d);

    if (b > 0.0) {
      return {(-b - sqrt_d) / (2.0 * a), 2.0 * c / (-b - sqrt_d)};
    }
    if (b < 0.0) {
      return {2.0 * c / (-b + sqrt_d), (-b + sqrt_d) / (2.0 * a)};
    }
    return {(-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a)};
  }
};

}  // namespace polatory::isosurface
