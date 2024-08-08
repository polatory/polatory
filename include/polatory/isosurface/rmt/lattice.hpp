#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/rmt/neighbor.hpp>
#include <polatory/isosurface/rmt/node.hpp>
#include <polatory/isosurface/rmt/node_list.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::isosurface::rmt {

class lattice : public primitive_lattice {
  using Base = primitive_lattice;
  using Node = node;
  using NodeList = node_list;

  static constexpr double kVertexPositionMinimumOffset = 1e-6;
  static constexpr double kZeroValueReplacement = 1e-100;

 public:
  lattice(const geometry::bbox3d& bbox, double resolution, const geometry::matrix3d& aniso)
      : Base(bbox, resolution, aniso) {}

  // Add all nodes within the second extended bbox.
  void add_all_nodes(const field_function& field_fn, double isovalue) {
    std::vector<cell_vector> nodes;
    std::vector<cell_vector> new_nodes;

    auto [cv2_min, cv2_max] = third_cell_vector_range();
    for (auto cv2 = cv2_min; cv2 <= cv2_max; cv2++) {
      auto [cv1_min, cv1_max] = second_cell_vector_range(cv2);
      for (auto cv1 = cv1_min; cv1 <= cv1_max; cv1++) {
        auto [cv0_min, cv0_max] = first_cell_vector_range(cv1, cv2);
        for (auto cv0 = cv0_min; cv0 <= cv0_max; cv0++) {
          cell_vector cv(cv0, cv1, cv2);
          if (add_node_unchecked(cv)) {
            new_nodes.push_back(cv);
          }
        }
      }

      evaluate_field(field_fn, isovalue);
      generate_vertices(new_nodes);
      remove_free_nodes(nodes);
      if (cv2 == cv2_max) {
        remove_free_nodes(new_nodes);
      }

      nodes.swap(new_nodes);
      new_nodes.clear();
    }

    update_neighbor_cache();
  }

  bool is_boundary_node(const cell_vector& cv) const {
    for (edge_index ei = 0; ei < 14; ei++) {
      geometry::vector3d p = cell_node_point(neighbor(cv, ei));
      if (!second_extended_bbox().contains(p)) {
        return true;
      }
    }
    return false;
  }

  void add_nodes_from_seed_points(geometry::points3d seed_points, const field_function& field_fm,
                                  double isovalue) {
    std::vector<seed_node> seed_nodes;

    for (auto seed_point : seed_points.rowwise()) {
      const auto& min = bbox().min();
      const auto& max = bbox().max();
      geometry::point3d clamped = seed_point.array().max(min.array()).min(max.array());

      auto cv = closest_cell_vector(clamped);
      add_node(cv);
      for (const auto& ncv : knn_nodes(cv, 1)) {
        add_node(ncv);
      }
      seed_nodes.emplace_back(cv, geometry::vector3d::Zero(), 1);

      evaluate_field(field_fm, isovalue);
    }

    std::vector<cell_vector_pair> cv_pairs;

    std::vector<seed_node> new_seed_nodes;
    while (!seed_nodes.empty()) {
      for (const auto& seed : seed_nodes) {
        const auto& cv = seed.cv;
        const auto& corrector = seed.corrector;
        auto k = seed.k;
        const auto& n = node_list_.at(cv);

        std::vector<cell_vector> ncvs;
        auto found_intersection = false;
        for (const auto& ncv : knn_nodes(cv, k)) {
          auto boundary_node = is_boundary_node(ncv);

          const auto& nn = node_list_.at(ncv);
          if (k == 1 && nn.value_sign() != n.value_sign()) {
            // The following usage of the if statement maximizes the chances of successful
            // surface tracking.
            cv_pairs.push_back(make_cell_vector_pair(cv, ncv));
            if (!boundary_node) {
              found_intersection = true;
              break;
            }
          }

          if (boundary_node) {
            // The gradient cannot be computed at a boundary node.
            continue;
          }

          if (std::abs(nn.value()) < std::abs(n.value())) {
            ncvs.push_back(ncv);
          }
        }

        if (found_intersection) {
          continue;
        }

        if (ncvs.empty()) {
          if (k >= 10) {
            // Give up.
            continue;
          }
          auto nn_cvs = knn_nodes(cv, k + 1);
          if (nn_cvs.empty()) {
            // No more nodes in the second extended bbox.
            continue;
          }
          for (const auto& nncv : nn_cvs) {
            add_node(nncv);
          }
          new_seed_nodes.emplace_back(cv, corrector, k + 1);
          continue;
        }

        geometry::vector3d neg_grad = -gradient(cv).normalized();

        auto ncv_it = std::min_element(
            ncvs.begin(), ncvs.end(),
            [this, &n, &neg_grad, &corrector](const auto& cva, const auto& cvb) {
              const auto& na = node_list_.at(cva);
              const auto& nb = node_list_.at(cvb);
              const auto& p = n.position();
              const auto& pa = na.position();
              const auto& pb = nb.position();
              geometry::vector3d va = pa - p;
              geometry::vector3d vb = pb - p;
              geometry::vector3d corrector_a = corrector + va - va.dot(neg_grad) * neg_grad;
              geometry::vector3d corrector_b = corrector + vb - vb.dot(neg_grad) * neg_grad;
              return corrector_a.norm() < corrector_b.norm();
            });

        {
          const auto& ncv = *ncv_it;
          for (const auto& nncv : knn_nodes(ncv, 1)) {
            add_node(nncv);
          }

          const auto& nn = node_list_.at(ncv);
          geometry::vector3d v = nn.position() - n.position();
          new_seed_nodes.emplace_back(ncv, corrector + v - v.dot(neg_grad) * neg_grad, 1);
        }
      }

      std::swap(seed_nodes, new_seed_nodes);
      new_seed_nodes.clear();

      evaluate_field(field_fm, isovalue);
    }

    std::unordered_set<cell_vector_pair, cell_vector_pair_hash> visited_cv_pairs;
    std::vector<cell_vector_pair> new_cv_pairs;
    while (!cv_pairs.empty()) {
      for (const auto& pair : cv_pairs) {
        if (visited_cv_pairs.contains(pair)) {
          continue;
        }
        visited_cv_pairs.insert(pair);

        const auto& [cv0, cv1] = pair;
        const auto& n0 = node_list_.at(cv0);
        const auto& n1 = node_list_.at(cv1);
        if (n0.value_sign() == n1.value_sign()) {
          continue;
        }

        std::vector<cell_vector> cv0_neighbors;
        std::vector<cell_vector> cv1_neighbors;
        for (const auto& ncv : knn_nodes(cv0, 1)) {
          cv0_neighbors.push_back(ncv);
        }
        for (const auto& ncv : knn_nodes(cv1, 1)) {
          cv1_neighbors.push_back(ncv);
        }
        std::sort(cv0_neighbors.begin(), cv0_neighbors.end(), cell_vector_less());
        std::sort(cv1_neighbors.begin(), cv1_neighbors.end(), cell_vector_less());
        std::vector<cell_vector> common_neighbors;
        std::set_intersection(cv0_neighbors.begin(), cv0_neighbors.end(), cv1_neighbors.begin(),
                              cv1_neighbors.end(), std::back_inserter(common_neighbors),
                              cell_vector_less());

        for (const auto& cv : common_neighbors) {
          add_node(cv);
          if (node_list_.contains(cv)) {
            new_cv_pairs.push_back(make_cell_vector_pair(cv0, cv));
            new_cv_pairs.push_back(make_cell_vector_pair(cv1, cv));
          }
        }
      }

      std::swap(cv_pairs, new_cv_pairs);
      new_cv_pairs.clear();

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

  void clear() {
    node_list_.clear();
    nodes_to_evaluate_.clear();
    value_at_arbitrary_point_.reset();
    vertices_.clear();
    vertices_to_refine_.clear();
    cluster_map_.clear();
  }

  void cluster_vertices() {
    const auto& min = first_extended_bbox().min();
    const auto& max = first_extended_bbox().max();

    for (auto& cv_node : node_list_) {
      auto& node = cv_node.second;
      const auto& p = node.position();
      if ((p.array() <= min.array() || p.array() >= max.array()).any()) {
        // Do not cluster vertices of a node on/outside the bbox,
        // as this can produce a boundary vertex inside the bbox.
        continue;
      }
      node.cluster(vertices_, cluster_map_);
    }
  }

  void generate_vertices(const std::vector<cell_vector>& node_cvs) {
#pragma omp parallel for
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (std::size_t i = 0; i < node_cvs.size(); i++) {
      const auto& cv0 = node_cvs.at(i);
      auto& node0 = node_list_.at(cv0);
      auto v0 = node0.value();

      for (edge_index ei = 0; ei < 14; ei++) {
        auto cv1 = neighbor(cv0, ei);
        if (!cell_vector_less()(cv1, cv0)) {
          continue;
        }

        auto* node1_ptr = node_list_.node_ptr(cv1);
        if (node1_ptr == nullptr) {
          // There is no neighbor node on the opposite end of the edge.
          continue;
        }

        auto& node1 = *node1_ptr;
        auto v1 = node1.value();

        if (v0 * v1 > 0.0) {
          // There is no intersection on the edge.
          continue;
        }

        auto t = v0 / (v0 - v1);

#pragma omp critical
        {
          auto vi = static_cast<vertex_index>(vertices_.size());
          auto opp_ei = kOppositeEdge.at(ei);

          if (t < 0.5) {
            node0.insert_vertex(vi, ei);
            vertices_to_refine_.emplace_back(cv0, ei, vi,                                  //
                                             0.0, v0,                                      //
                                             t, std::numeric_limits<double>::quiet_NaN(),  //
                                             1.0, v1);
          } else {
            node1.insert_vertex(vi, opp_ei);
            vertices_to_refine_.emplace_back(cv1, opp_ei, vi,                                    //
                                             0.0, v1,                                            //
                                             1.0 - t, std::numeric_limits<double>::quiet_NaN(),  //
                                             1.0, v0);
          }

          node0.set_intersection(ei);
          node1.set_intersection(opp_ei);

          const auto& v = vertices_to_refine_.back();
          vertices_.emplace_back(v.position_clamped(node_list_));
        }
      }
    }
  }

  geometry::points3d get_vertices() const {
    geometry::points3d vertices(static_cast<index_t>(vertices_.size()), 3);
    auto it = vertices.rowwise().begin();
    for (const auto& v : vertices_) {
      *it++ = v;
    }
    return vertices;
  }

  void refine_vertices(const field_function& field_fn, double isovalue, int num_passes) {
    if (num_passes <= 0) {
      return;
    }

    auto n = static_cast<index_t>(vertices_to_refine_.size());
    geometry::points3d vertices(n, 3);

    for (auto pass = 0; pass < num_passes; pass++) {
      for (index_t i = 0; i < n; i++) {
        const auto& v = vertices_to_refine_.at(i);
        vertices.row(i) = v.position_unclamped(node_list_);
      }

      vectord vertex_values = field_fn(vertices).array() - isovalue;
      vertex_values = (vertex_values.array() == 0.0).select(kZeroValueReplacement, vertex_values);

      for (index_t i = 0; i < n; i++) {
        auto& v = vertices_to_refine_.at(i);
        v.v1 = vertex_values(i);
      }

      refine_vertices(vertices_to_refine_);
    }

    for (index_t i = 0; i < n; i++) {
      const auto& v = vertices_to_refine_.at(i);
      vertices_.at(v.vi) = v.position_clamped(node_list_);

      if (v.t1 >= 0.5) {
        auto& node0 = node_list_.at(v.node_cv);
        auto& node1 = node_list_.at(neighbor(v.node_cv, v.ei));
        node0.remove_vertex(v.ei);
        node1.insert_vertex(v.vi, kOppositeEdge.at(v.ei));
      }
    }
  }

  index_t uncluster_vertices(const std::unordered_set<vertex_index>& vis) {
    index_t num_unclustered = 0;

    auto it = cluster_map_.begin();
    while (it != cluster_map_.end()) {
      if (vis.contains(it->second)) {
        // Uncluster.
        it = cluster_map_.erase(it);
        ++num_unclustered;
      } else {
        ++it;
      }
    }

    return num_unclustered;
  }

  double value_at_arbitrary_point() const { return value_at_arbitrary_point_.value(); }

 private:
  friend class surface_generator;

  struct seed_node {
    cell_vector cv;
    geometry::vector3d corrector;
    int k;
  };

  struct vertex_to_refine {
    cell_vector node_cv;
    edge_index ei{};
    vertex_index vi{};
    double t0{};
    double v0{};
    double t1{};
    double v1{};
    double t2{};
    double v2{};

    geometry::point3d position_clamped(const NodeList& node_list) const {
      const auto& node0 = node_list.at(node_cv);
      const auto& node1 = node_list.at(neighbor(node_cv, ei));
      const auto& p0 = node0.position();
      const auto& p1 = node1.position();
      auto t = std::clamp(t1, kVertexPositionMinimumOffset, 1.0 - kVertexPositionMinimumOffset);
      return p0 + t * (p1 - p0);
    }

    geometry::point3d position_unclamped(const NodeList& node_list) const {
      const auto& node0 = node_list.at(node_cv);
      const auto& node1 = node_list.at(neighbor(node_cv, ei));
      const auto& p0 = node0.position();
      const auto& p1 = node1.position();
      auto t = t1;
      return p0 + t * (p1 - p0);
    }
  };

  // Returns true if the node is added.
  bool add_node(const cell_vector& cv) {
    if (node_list_.contains(cv)) {
      return false;
    }

    return add_node_unchecked(cv);
  }

  // Returns true if the node is added.
  bool add_node_unchecked(const cell_vector& cv) {
    auto p = cell_node_point(cv);

    if (!second_extended_bbox().contains(p)) {
      return false;
    }

    // To prevent surface clipping from generating tiny faces, which can cause
    // various numerical issues, snap the point to the bbox if it is very close to it.

    const auto& min = bbox().min();
    const auto& max = bbox().max();
    auto tiny = 1e-10 * resolution();

    p = ((p.array() - min.array()).abs() < tiny).select(min, p);
    p = ((p.array() - max.array()).abs() < tiny).select(max, p);

    node_list_.emplace(cv, Node(p));

    nodes_to_evaluate_.push_back(cv);
    return true;
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

    vectord values = field_fn(points).array() - isovalue;
    values = (values.array() == 0.0).select(kZeroValueReplacement, values);
    value_at_arbitrary_point_.emplace(values(0));

    index_t i{};
    for (const auto& cv : nodes_to_evaluate_) {
      auto value = values(i);
      node_list_.at(cv).set_value(value);
      i++;
    }

    nodes_to_evaluate_.clear();
  }

  geometry::vector3d gradient(const auto& cv) const {
    std::array<cell_vector, 7> cvs{cv,
                                   // 6NN
                                   neighbor(cv, edge::k1), neighbor(cv, edge::k3),
                                   neighbor(cv, edge::k5), neighbor(cv, edge::k8),
                                   neighbor(cv, edge::kA), neighbor(cv, edge::kC)};
    matrixd a(7, 4);
    vectord b(7);
    for (index_t i = 0; i < 7; i++) {
      const auto& n = node_list_.at(cvs.at(i));
      a.row(i) << n.position().x(), n.position().y(), n.position().z(), 1.0;
      b(i) = std::abs(n.value());
    }

    matrixd system = a.transpose() * a;
    vectord rhs = a.transpose() * b;
    vectord x = system.ldlt().solve(rhs);
    return x.head<3>();
  }

  static bool has_intersection(const Node* a, const Node* b) {
    return a != nullptr && b != nullptr && a->value_sign() != b->value_sign();
  }

  std::vector<cell_vector> knn_nodes(const cell_vector& cv, int k) const {
    std::unordered_set<cell_vector, cell_vector_hash> visited;
    visited.insert(cv);

    std::vector<cell_vector> frontier{cv};
    std::vector<cell_vector> next_frontier;

    for (int i = 0; i < k; i++) {
      for (const auto& ncv : frontier) {
        for (edge_index ei = 0; ei < 14; ei++) {
          auto nncv = neighbor(ncv, ei);
          if (visited.contains(nncv)) {
            continue;
          }
          auto p = cell_node_point(nncv);
          if (!second_extended_bbox().contains(p)) {
            continue;
          }
          visited.insert(nncv);
          next_frontier.push_back(nncv);
        }
      }

      std::swap(frontier, next_frontier);
      next_frontier.clear();
    }

    return frontier;
  }

  static void refine_vertices(std::vector<vertex_to_refine>& vertices_to_refine) {
    for (auto& v : vertices_to_refine) {
      // Solve y = a x^2 + b x + c for a, b, c with (x, y) = (t0, v0), (t1, v1), (t2, v2).
      Eigen::Matrix3d a;
      a << v.t0 * v.t0, v.t0, 1.0, v.t1 * v.t1, v.t1, 1.0, v.t2 * v.t2, v.t2, 1.0;
      Eigen::Vector3d b(v.v0, v.v1, v.v2);
      Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr_a(a);
      if (!qr_a.isInvertible()) {
        continue;
      }
      Eigen::Vector3d c = qr_a.solve(b);

      // Solve a x^2 + b x + c = 0 for x, where 0 < x < 1.
      auto [s0, s1] = solve_quadratic(c(0), c(1), c(2));
      auto s = v.t0 < s0 && s0 < v.t2 ? s0 : s1;

      if (s < v.t1) {
        // (t0', t1', t2') = (t0, s, t1).
        v.t2 = v.t1;
        v.v2 = v.v1;
      } else {
        // (t0', t1', t2') = (t1, s, t2).
        v.t0 = v.t1;
        v.v0 = v.v1;
      }

      v.t1 = s;
      v.v1 = std::numeric_limits<double>::quiet_NaN();
    }
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

  void update_neighbor_cache() {
    for (auto& cv_node : node_list_) {
      const auto& cv = cv_node.first;
      auto& node = cv_node.second;

      auto neighbors = std::make_unique<std::array<Node*, 14>>();

      for (edge_index ei = 0; ei < 14; ei++) {
        neighbors->at(ei) = node_list_.node_ptr(neighbor(cv, ei));
      }

      node.set_neighbors(std::move(neighbors));
    }
  }

  NodeList node_list_;
  std::vector<cell_vector> nodes_to_evaluate_;
  std::optional<double> value_at_arbitrary_point_;
  std::vector<geometry::point3d> vertices_;
  std::vector<vertex_to_refine> vertices_to_refine_;
  std::unordered_map<vertex_index, vertex_index> cluster_map_;
};

}  // namespace polatory::isosurface::rmt
