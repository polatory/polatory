#pragma once

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

inline const std::array<edge_index, 14> kOppositeEdge{
    edge::k7, edge::k8, edge::k9, edge::kA, edge::kB, edge::kC, edge::kD,
    edge::k0, edge::k1, edge::k2, edge::k3, edge::k4, edge::k5, edge::k6};

class lattice : public primitive_lattice {
  using Base = primitive_lattice;
  using Node = node;
  using NodeList = node_list;

  static constexpr double kZeroValueReplacement = 1e-100;

 public:
  lattice(const geometry::bbox3d& bbox, double resolution, const geometry::matrix3d& aniso)
      : Base(bbox, resolution, aniso) {}

  // Add all nodes inside the boundary.
  void add_all_nodes(const field_function& field_fn, double isovalue) {
    auto ext_bbox_corners = extended_bbox().corners();

    cell_vectors cvs(8, 3);
    for (index_t i = 0; i < 8; i++) {
      cvs.row(i) = cell_vector_from_point(ext_bbox_corners.row(i));
    }

    // Bounds of cell vectors for enumerating all nodes in the extended bbox.
    cell_vector cv_min = cvs.colwise().minCoeff().array() + 1;
    cell_vector cv_max = cvs.colwise().maxCoeff();

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

  void add_cell_from_point(const geometry::point3d& p) {
    add_cell(cell_vector_from_point(clamp_to_bbox(p)));
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

  void clear() {
    node_list_.clear();
    nodes_to_evaluate_.clear();
    cluster_map_.clear();
    vertices_.clear();
  }

  void cluster_vertices() {
    const auto& min = bbox().min();
    const auto& max = bbox().max();

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
    static constexpr std::array<edge_index, 7> CellEdgeIndices{
        edge::k0, edge::k1, edge::k2, edge::k3, edge::k4, edge::k5, edge::k6};

#pragma omp parallel for
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (std::size_t i = 0; i < node_cvs.size(); i++) {
      const auto& cv0 = node_cvs.at(i);
      auto& node0 = node_list_.at(cv0);
      const auto& p0 = node0.position();
      auto v0 = node0.value();

      for (auto ei : CellEdgeIndices) {
        auto cv1 = neighbor(cv0, ei);
        auto* node1_ptr = node_list_.node_ptr(cv1);
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
    for (index_t i = 0; i < n; i++) {
      vertices.row(i) = vertices_.at(i);
    }

    for (auto pass = 0; pass < num_passes; pass++) {
      vectord vertex_values = field_fn(vertices).array() - isovalue;

      for (index_t i = 0; i < n; i++) {
        auto& v = vertices_to_refine_.at(i);
        v.v1 = vertex_values(i);
      }

      refine_vertices(vertices_to_refine_);

      for (index_t i = 0; i < n; i++) {
        const auto& v = vertices_to_refine_.at(i);
        const auto& node0 = node_list_.at(v.node_cv);
        const auto& node1 = node_list_.at(neighbor(v.node_cv, v.ei));
        const auto& p0 = node0.position();
        const auto& p1 = node1.position();
        vertices.row(i) = p0 + v.t1 * (p1 - p0);
      }
    }

    for (index_t i = 0; i < n; i++) {
      const auto& v = vertices_to_refine_.at(i);
      vertices_.at(v.vi) = vertices.row(i);

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

  double value_at_arbitrary_point() const { return value_at_arbitrary_point_; }

 private:
  friend class surface_generator;

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
  };

  // Add nodes corresponding to eight vertices of the cell.
  void add_cell(const cell_vector& cv) {
    if (added_cells_.contains(cv)) {
      return;
    }

    std::array<cell_vector, 8> node_cvs{cv,
                                        neighbor(cv, edge::k0),
                                        neighbor(cv, edge::k1),
                                        neighbor(cv, edge::k2),
                                        neighbor(cv, edge::k3),
                                        neighbor(cv, edge::k4),
                                        neighbor(cv, edge::k5),
                                        neighbor(cv, edge::k6)};

    for (const auto& node_cv : node_cvs) {
      add_node(node_cv);
    }

    added_cells_.emplace(cv);
    last_added_cells_.push_back(cv);
  }

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

    if (!extended_bbox().contains(p)) {
      return false;
    }

    node_list_.emplace(cv, Node{p});

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

    vectord values = field_fn(points).array() - isovalue;
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

  static bool has_intersection(const Node* a, const Node* b) {
    return a != nullptr && b != nullptr && a->value_sign() != b->value_sign();
  }

  // Returns true if all of the cell's vertices are in the extended bbox.
  bool is_interior_cell(const cell_vector& cv) {
    std::array<cell_vector, 8> node_cvs{cv,
                                        neighbor(cv, edge::k0),
                                        neighbor(cv, edge::k1),
                                        neighbor(cv, edge::k2),
                                        neighbor(cv, edge::k3),
                                        neighbor(cv, edge::k4),
                                        neighbor(cv, edge::k5),
                                        neighbor(cv, edge::k6)};

    return std::all_of(node_cvs.begin(), node_cvs.end(), [this](const auto& node_cv) {
      auto p = cell_node_point(node_cv);
      return extended_bbox().contains(p);
    });
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

  void track_surface() {
    std::vector<cell_vector> last_added_cells(std::move(last_added_cells_));
    last_added_cells_.clear();

    // Check 12 edges of each cell and add neighbor cells adjacent to an edge
    // at which ends the field values take opposite signs.
    for (const auto& cv : last_added_cells) {
      auto iaaa = cv;
      auto ibaa = neighbor(cv, edge::k0);
      auto ibab = neighbor(cv, edge::k1);
      auto iaab = neighbor(cv, edge::k2);
      auto ibba = neighbor(cv, edge::k3);
      auto ibbb = neighbor(cv, edge::k4);
      auto iabb = neighbor(cv, edge::k5);
      auto iaba = neighbor(cv, edge::k6);

      const auto* aaa = node_list_.node_ptr(iaaa);
      const auto* baa = node_list_.node_ptr(ibaa);
      const auto* bab = node_list_.node_ptr(ibab);
      const auto* aab = node_list_.node_ptr(iaab);
      const auto* bba = node_list_.node_ptr(ibba);
      const auto* bbb = node_list_.node_ptr(ibbb);
      const auto* abb = node_list_.node_ptr(iabb);
      const auto* aba = node_list_.node_ptr(iaba);

      auto found_intersection = false;

      // axx and bxx
      if (has_intersection(aaa, baa)) {  // o -> 0
        add_cell(neighbor(iaaa, edge::kD));
        add_cell(neighbor(iaaa, edge::kC));
        add_cell(neighbor(iaaa, edge::k9));
        found_intersection = true;
      }
      if (has_intersection(aab, bab)) {  // 2 -> 1
        add_cell(iaab);
        add_cell(neighbor(iaab, edge::kD));
        add_cell(neighbor(iaab, edge::kC));
        found_intersection = true;
      }
      if (has_intersection(aba, bba)) {  // 6 -> 3
        add_cell(iaba);
        add_cell(neighbor(iaba, edge::kC));
        add_cell(neighbor(iaba, edge::k9));
        found_intersection = true;
      }
      if (has_intersection(abb, bbb)) {  // 5 -> 4
        add_cell(iabb);
        add_cell(neighbor(iabb, edge::kD));
        add_cell(neighbor(iabb, edge::k9));
        found_intersection = true;
      }

      // xax and xbx
      if (has_intersection(aaa, aba)) {  // o -> 6
        add_cell(neighbor(iaaa, edge::k7));
        add_cell(neighbor(iaaa, edge::k8));
        add_cell(neighbor(iaaa, edge::k9));
        found_intersection = true;
      }
      if (has_intersection(aab, abb)) {  // 2 -> 5
        add_cell(iaab);
        add_cell(neighbor(iaab, edge::k7));
        add_cell(neighbor(iaab, edge::k8));
        found_intersection = true;
      }
      if (has_intersection(baa, bba)) {  // 0 -> 3
        add_cell(ibaa);
        add_cell(neighbor(ibaa, edge::k8));
        add_cell(neighbor(ibaa, edge::k9));
        found_intersection = true;
      }
      if (has_intersection(bab, bbb)) {  // 1 -> 4
        add_cell(ibab);
        add_cell(neighbor(ibab, edge::k7));
        add_cell(neighbor(ibab, edge::k9));
        found_intersection = true;
      }

      // xxa and xxb
      if (has_intersection(aaa, aab)) {  // o -> 2
        add_cell(neighbor(iaaa, edge::k7));
        add_cell(neighbor(iaaa, edge::kA));
        add_cell(neighbor(iaaa, edge::kD));
        found_intersection = true;
      }
      if (has_intersection(baa, bab)) {  // 0 -> 1
        add_cell(ibaa);
        add_cell(neighbor(ibaa, edge::kA));
        add_cell(neighbor(ibaa, edge::kD));
        found_intersection = true;
      }
      if (has_intersection(aba, abb)) {  // 6 -> 5
        add_cell(iaba);
        add_cell(neighbor(iaba, edge::k7));
        add_cell(neighbor(iaba, edge::kA));
        found_intersection = true;
      }
      if (has_intersection(bba, bbb)) {  // 3 -> 4
        add_cell(ibba);
        add_cell(neighbor(ibba, edge::k7));
        add_cell(neighbor(ibba, edge::kD));
        found_intersection = true;
      }

      if (found_intersection) {
        continue;
      }

      // Descend along the gradient.

      std::array<const Node*, 8> nodes{aaa, baa, aba, bba, aab, bab, abb, bbb};
      if (std::any_of(nodes.begin(), nodes.end(), [](const auto* n) { return n == nullptr; })) {
        continue;
      }

      // Vertex-neighbor cells.
      {
        std::array<double, 8> values{
            std::abs(aaa->value()),  // aaa
            std::abs(baa->value()),  // baa
            std::abs(aba->value()),  // aba
            std::abs(bba->value()),  // bba
            std::abs(aab->value()),  // aab
            std::abs(bab->value()),  // bab
            std::abs(abb->value()),  // abb
            std::abs(bbb->value()),  // bbb
        };
        std::array<cell_vector, 8> neighbors{
            cv + cell_vector(-1, -1, -1),  // aaa
            cv + cell_vector(1, -1, -1),   // baa
            cv + cell_vector(-1, 1, -1),   // aba
            cv + cell_vector(1, 1, -1),    // bba
            cv + cell_vector(-1, -1, 1),   // aab
            cv + cell_vector(1, -1, 1),    // bab
            cv + cell_vector(-1, 1, 1),    // abb
            cv + cell_vector(1, 1, 1),     // bbb
        };
        std::array<bool, 8> feasible{};
        std::transform(neighbors.begin(), neighbors.end(), feasible.begin(),
                       [this](const auto& neighbor) { return is_interior_cell(neighbor); });

        std::array<int, 8> indices{0, 1, 2, 3, 4, 5, 6, 7};
        std::sort(indices.begin(), indices.end(), [&values, &feasible](auto i, auto j) {
          return values.at(i) != values.at(j) ? values.at(i) < values.at(j)
                                              : feasible.at(i) < feasible.at(j);
        });
        auto begin = std::find_if(indices.begin(), indices.end(),
                                  [&feasible](auto i) { return feasible.at(i); });
        auto end = std::upper_bound(begin, indices.end(), *begin, [&values](auto i, auto j) {
          return values.at(i) < values.at(j);
        });

        if (std::distance(indices.begin(), begin) < 4) {
          for (auto it = begin; it != end; ++it) {
            auto neighbor = neighbors.at(*it);
            add_cell(neighbor);
          }
          continue;
        }
      }

      // Edge-neighbor cells.
      {
        std::array<double, 12> values{
            std::abs(aaa->value() + aab->value()),  // aax
            std::abs(aba->value() + abb->value()),  // abx
            std::abs(baa->value() + bab->value()),  // bax
            std::abs(bba->value() + bbb->value()),  // bbx
            std::abs(aaa->value() + aba->value()),  // axa
            std::abs(aab->value() + abb->value()),  // axb
            std::abs(baa->value() + bba->value()),  // bxa
            std::abs(bab->value() + bbb->value()),  // bxb
            std::abs(aaa->value() + baa->value()),  // xaa
            std::abs(aab->value() + bab->value()),  // xab
            std::abs(aba->value() + bba->value()),  // xba
            std::abs(abb->value() + bbb->value()),  // xbb
        };
        std::array<cell_vector, 12> neighbors{
            cv + cell_vector(-1, -1, 0),  // aax
            cv + cell_vector(-1, 1, 0),   // abx
            cv + cell_vector(1, -1, 0),   // bax
            cv + cell_vector(1, 1, 0),    // bbx
            cv + cell_vector(-1, 0, -1),  // axa
            cv + cell_vector(-1, 0, 1),   // axb
            cv + cell_vector(1, 0, -1),   // bxa
            cv + cell_vector(1, 0, 1),    // bxb
            cv + cell_vector(0, -1, -1),  // xaa
            cv + cell_vector(0, -1, 1),   // xab
            cv + cell_vector(0, 1, -1),   // xba
            cv + cell_vector(0, 1, 1),    // xbb
        };
        std::array<bool, 12> feasible{};
        std::transform(neighbors.begin(), neighbors.end(), feasible.begin(),
                       [this](const auto& neighbor) { return is_interior_cell(neighbor); });

        std::array<int, 12> indices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::sort(indices.begin(), indices.end(), [&values, &feasible](auto i, auto j) {
          return values.at(i) != values.at(j) ? values.at(i) < values.at(j)
                                              : feasible.at(i) < feasible.at(j);
        });
        auto begin = std::find_if(indices.begin(), indices.end(),
                                  [&feasible](auto i) { return feasible.at(i); });
        auto end = std::upper_bound(begin, indices.end(), *begin, [&values](auto i, auto j) {
          return values.at(i) < values.at(j);
        });

        if (std::distance(indices.begin(), begin) < 6) {
          for (auto it = begin; it != end; ++it) {
            auto neighbor = neighbors.at(*it);
            add_cell(neighbor);
          }
          continue;
        }
      }

      // Face-neighbor cells.
      {
        std::array<double, 6> values{
            std::abs(aaa->value() + aab->value() + aba->value() + abb->value()),  // axx
            std::abs(baa->value() + bab->value() + bba->value() + bbb->value()),  // bxx
            std::abs(aaa->value() + aab->value() + baa->value() + bab->value()),  // xax
            std::abs(aba->value() + abb->value() + bba->value() + bbb->value()),  // xbx
            std::abs(aaa->value() + baa->value() + aba->value() + bba->value()),  // xxa
            std::abs(aab->value() + bab->value() + abb->value() + bbb->value()),  // xxb
        };
        std::array<cell_vector, 6> neighbors{
            cv + cell_vector(-1, 0, 0),  // axx
            cv + cell_vector(1, 0, 0),   // bxx
            cv + cell_vector(0, -1, 0),  // xax
            cv + cell_vector(0, 1, 0),   // xbx
            cv + cell_vector(0, 0, -1),  // xxa
            cv + cell_vector(0, 0, 1),   // xxb
        };
        std::array<bool, 6> feasible{};
        std::transform(neighbors.begin(), neighbors.end(), feasible.begin(),
                       [this](const auto& neighbor) { return is_interior_cell(neighbor); });

        std::array<int, 6> indices{0, 1, 2, 3, 4, 5};
        std::sort(indices.begin(), indices.end(), [&values, &feasible](auto i, auto j) {
          return values.at(i) != values.at(j) ? values.at(i) < values.at(j)
                                              : feasible.at(i) < feasible.at(j);
        });
        auto begin = std::find_if(indices.begin(), indices.end(),
                                  [&feasible](auto i) { return feasible.at(i); });
        auto end = std::upper_bound(begin, indices.end(), *begin, [&values](auto i, auto j) {
          return values.at(i) < values.at(j);
        });

        if (std::distance(indices.begin(), begin) < 3) {
          for (auto it = begin; it != end; ++it) {
            const auto& neighbor = neighbors.at(*it);
            add_cell(neighbor);
          }
        }
      }
    }
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
  std::unordered_set<cell_vector, cell_vector_hash> added_cells_;
  std::vector<cell_vector> last_added_cells_;
  double value_at_arbitrary_point_{kZeroValueReplacement};
  std::vector<geometry::point3d> vertices_;
  std::vector<vertex_to_refine> vertices_to_refine_;
  std::unordered_map<vertex_index, vertex_index> cluster_map_;
};

}  // namespace polatory::isosurface::rmt
