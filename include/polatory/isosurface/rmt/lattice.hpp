#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/rmt/neighbor.hpp>
#include <polatory/isosurface/rmt/node.hpp>
#include <polatory/isosurface/rmt/node_list.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/isosurface/rmt/tetrahedron.hpp>
#include <polatory/isosurface/sign.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::isosurface::rmt {

class Lattice : public PrimitiveLattice {
  using Base = PrimitiveLattice;

  static constexpr double kVertexPositionMinimumOffset = 1e-4;

 public:
  Lattice(const geometry::Bbox3& bbox, double resolution, const Mat3& aniso)
      : Base(bbox, resolution, aniso) {}

  // Add all nodes within the second extended bbox.
  void add_all_nodes(const FieldFunction& field_fn, double isovalue) {
    value_at_arbitrary_point_.emplace(bbox().center(), *this);

    std::vector<LatticeCoordinates> nodes;
    std::vector<LatticeCoordinates> new_nodes;

    auto [lc2_min, lc2_max] = third_lattice_coordinate_range();
    for (auto lc2 = lc2_min; lc2 <= lc2_max; lc2++) {
      auto [lc1_min, lc1_max] = second_lattice_coordinate_range(lc2);
      for (auto lc1 = lc1_min; lc1 <= lc1_max; lc1++) {
        auto [lc0_min, lc0_max] = first_lattice_coordinate_range(lc1, lc2);
        for (auto lc0 = lc0_min; lc0 <= lc0_max; lc0++) {
          LatticeCoordinates lc(lc0, lc1, lc2);
          if (add_node_unchecked(lc)) {
            new_nodes.push_back(lc);
          }
        }
      }

      evaluate_field(field_fn, isovalue);
      generate_vertices(new_nodes);
      remove_free_nodes(nodes);
      if (lc2 == lc2_max) {
        remove_free_nodes(new_nodes);
      }

      nodes.swap(new_nodes);
      new_nodes.clear();
    }
  }

  void add_nodes_from_seed_points(const geometry::Points3& seed_points,
                                  const FieldFunction& field_fm, double isovalue) {
    const auto& min = bbox().min();
    const auto& max = bbox().max();

    value_at_arbitrary_point_.emplace(seed_points.row(0).array().max(min.array()).min(max.array()),
                                      *this);

    std::vector<Seed> seeds;

    for (auto seed_point : seed_points.rowwise()) {
      geometry::Point3 clamped = seed_point.array().max(min.array()).min(max.array());

      auto lc = lattice_coordinates_rounded(clamped);
      add_node(lc);
      for (const auto& nlc : knn_nodes(lc, 1)) {
        add_node(nlc);
      }
      seeds.emplace_back(lc, geometry::Vector3::Zero(), 1);

      evaluate_field(field_fm, isovalue);
    }

    std::sort(seeds.begin(), seeds.end(),
              [](const auto& a, const auto& b) { return LatticeCoordinatesLess()(a.lc, b.lc); });
    seeds.erase(std::unique(seeds.begin(), seeds.end(),
                            [](const auto& a, const auto& b) { return a.lc == b.lc; }),
                seeds.end());

    std::vector<LatticeCoordinatesPair> pairs;

    std::vector<Seed> new_seeds;
    std::vector<LatticeCoordinates> nlcs;
    while (!seeds.empty()) {
      for (const auto& seed : seeds) {
        const auto& lc = seed.lc;
        const auto& corrector = seed.corrector;
        auto k = seed.k;
        const auto& n = node_list_.at(lc);

        nlcs.clear();
        auto found_intersection = false;
        for (const auto& nlc : knn_nodes(lc, k)) {
          auto boundary_node = is_boundary_node(nlc);

          const auto& nn = node_list_.at(nlc);
          if (k == 1 && nn.value_sign() != n.value_sign()) {
            // The following usage of the if statement maximizes the chances of successful
            // surface tracking.
            pairs.push_back(make_lattice_coordinates_pair(lc, nlc));
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
            nlcs.push_back(nlc);
          }
        }

        if (found_intersection) {
          continue;
        }

        if (nlcs.empty()) {
          if (k >= 10) {
            // Give up.
            continue;
          }
          auto nnlcs = knn_nodes(lc, k + 1);
          if (nnlcs.empty()) {
            // No more nodes in the second extended bbox.
            continue;
          }
          for (const auto& nnlc : nnlcs) {
            add_node(nnlc);
          }
          new_seeds.emplace_back(lc, corrector, k + 1);
          continue;
        }

        geometry::Vector3 neg_grad = -gradient(lc).normalized();

        auto nlc_it = std::min_element(
            nlcs.begin(), nlcs.end(),
            [this, &n, &neg_grad, &corrector](const auto& lca, const auto& lcb) {
              const auto& na = node_list_.at(lca);
              const auto& nb = node_list_.at(lcb);
              const auto& p = n.position();
              const auto& pa = na.position();
              const auto& pb = nb.position();
              geometry::Vector3 va = pa - p;
              geometry::Vector3 vb = pb - p;
              geometry::Vector3 corrector_a = corrector + va - va.dot(neg_grad) * neg_grad;
              geometry::Vector3 corrector_b = corrector + vb - vb.dot(neg_grad) * neg_grad;
              return corrector_a.norm() < corrector_b.norm();
            });

        {
          const auto& nlc = *nlc_it;
          for (const auto& nnlc : knn_nodes(nlc, 1)) {
            add_node(nnlc);
          }

          const auto& nn = node_list_.at(nlc);
          geometry::Vector3 v = nn.position() - n.position();
          new_seeds.emplace_back(nlc, corrector + v - v.dot(neg_grad) * neg_grad, 1);
        }
      }

      std::swap(seeds, new_seeds);
      new_seeds.clear();

      evaluate_field(field_fm, isovalue);
    }

    std::unordered_set<LatticeCoordinatesPair, LatticeCoordinatesPairHash> visited_pairs;
    std::vector<LatticeCoordinatesPair> new_pairs;
    std::vector<LatticeCoordinates> neighbors0;
    std::vector<LatticeCoordinates> neighbors1;
    std::vector<LatticeCoordinates> common_neighbors;
    while (!pairs.empty()) {
      for (const auto& pair : pairs) {
        if (visited_pairs.contains(pair)) {
          continue;
        }
        visited_pairs.insert(pair);

        const auto& [lc0, lc1] = pair;
        const auto& n0 = node_list_.at(lc0);
        const auto& n1 = node_list_.at(lc1);
        if (n0.value_sign() == n1.value_sign()) {
          continue;
        }

        neighbors0.clear();
        neighbors1.clear();
        common_neighbors.clear();
        for (const auto& nlc : knn_nodes(lc0, 1)) {
          neighbors0.push_back(nlc);
        }
        for (const auto& nlc : knn_nodes(lc1, 1)) {
          neighbors1.push_back(nlc);
        }
        std::sort(neighbors0.begin(), neighbors0.end(), LatticeCoordinatesLess());
        std::sort(neighbors1.begin(), neighbors1.end(), LatticeCoordinatesLess());
        std::set_intersection(neighbors0.begin(), neighbors0.end(), neighbors1.begin(),
                              neighbors1.end(), std::back_inserter(common_neighbors),
                              LatticeCoordinatesLess());

        for (const auto& nlc : common_neighbors) {
          add_node(nlc);
          if (node_list_.contains(nlc)) {
            new_pairs.push_back(make_lattice_coordinates_pair(lc0, nlc));
            new_pairs.push_back(make_lattice_coordinates_pair(lc1, nlc));
          }
        }
      }

      std::swap(pairs, new_pairs);
      new_pairs.clear();

      evaluate_field(field_fm, isovalue);
    }

    std::vector<LatticeCoordinates> all_nodes;
    for (const auto& lc_node : node_list_) {
      all_nodes.push_back(lc_node.first);
    }

    generate_vertices(all_nodes);
    remove_free_nodes(all_nodes);
  }

  void clear() {
    node_list_.clear();
    nodes_to_evaluate_.clear();
    vertices_.clear();
    cluster_map_.clear();
    clustered_vertices_.clear();
    value_at_arbitrary_point_.reset();
  }

  void cluster_vertices() {
    std::vector<geometry::Point3> vertices(vertices_.size());
    for (std::size_t i = 0; i < vertices.size(); i++) {
      // Use the clamped positions to reduce the risk of generating overlapping faces
      // when there are multiple surfaces around the node.
      vertices.at(i) = vertices_.at(i).position_clamped(node_list_);
    }

    const auto& min = first_extended_bbox().min();
    const auto& max = first_extended_bbox().max();

    for (auto& lc_node : node_list_) {
      auto& node = lc_node.second;
      const auto& p = node.position();
      if ((p.array() <= min.array() || p.array() >= max.array()).any()) {
        // Do not cluster vertices of a node on/outside the bbox,
        // as this can produce a boundary vertex inside the bbox.
        continue;
      }
      node.cluster(vertices, cluster_map_, clustered_vertices_);
    }
  }

  Mesh get_mesh() const {
    std::vector<Face> faces_v;
    auto inserter = std::back_inserter(faces_v);
    for (const auto& lc_node : node_list_) {
      const auto& lc = lc_node.first;
      for (TetrahedronIterator it(lc, node_list_); it.is_valid(); ++it) {
        it->get_faces(inserter);
      }
    }

    Faces faces(static_cast<Index>(faces_v.size()), 3);
    Index n_faces = 0;

    auto it = faces.rowwise().begin();
    for (const auto& face : faces_v) {
      auto v0 = clustered_vertex_index(face(0));
      auto v1 = clustered_vertex_index(face(1));
      auto v2 = clustered_vertex_index(face(2));

      if (v0 == v1 || v1 == v2 || v2 == v0) {
        // Degenerate face (due to vertex clustering).
        continue;
      }

      *it++ << v0, v1, v2;
      n_faces++;
    }

    faces.conservativeResize(n_faces, 3);

    return {get_vertices(), faces};
  }

  void refine_vertices(const FieldFunction& field_fn, double isovalue, int num_passes) {
    if (num_passes <= 0) {
      return;
    }

    auto n = static_cast<Index>(vertices_.size());
    geometry::Points3 vertices(n, 3);

    for (auto pass = 0; pass < num_passes; pass++) {
      for (Index i = 0; i < n; i++) {
        const auto& v = vertices_.at(i);
        vertices.row(i) = v.position_unclamped(node_list_);
      }

      VecX vertex_values = field_fn(vertices).array() - isovalue;

      for (Index i = 0; i < n; i++) {
        auto& v = vertices_.at(i);
        v.v1 = vertex_values(i);
      }

      refine_vertices(vertices_);
    }

    for (const auto& v : vertices_) {
      if (v.t1 >= 0.5) {
        auto& node0 = node_list_.at(v.node_lc);
        auto& node1 = node_list_.at(neighbor(v.node_lc, v.ei));
        node0.remove_vertex(v.ei);
        node1.insert_vertex(v.vi, kOppositeEdge.at(v.ei));
      }
    }
  }

  Index uncluster_vertices(const std::unordered_set<Index>& vis) {
    Index num_unclustered = 0;

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

  BinarySign value_sign_at_arbitrary_point_within_bbox() const {
    return value_at_arbitrary_point_.value().value_sign();
  }

 private:
  class InterpolatedValue {
   public:
    explicit InterpolatedValue(const geometry::Point3& p, const Base& lattice)
        : tet_(lattice.tetrahedron(p)),
          weights_(barycentric_coordinates(
              p, lattice.position(tet_.row(0)), lattice.position(tet_.row(1)),
              lattice.position(tet_.row(2)), lattice.position(tet_.row(3)))) {}

    void tell(const LatticeCoordinates& lc, double value) {
      for (Index i = 0; i < 4; i++) {
        if (tet_.row(i) == lc) {
          values_(i) = value;
          populated_.at(i) = true;
          break;
        }
      }
    }

    double value() const {
      if (std::any_of(populated_.begin(), populated_.end(), [](auto init) { return !init; })) {
        throw std::runtime_error("not all values are populated");
      }

      return weights_.dot(values_);
    }

    BinarySign value_sign() const { return sign(value()); }

   private:
    const Eigen::Matrix<int, 4, 3, Eigen::RowMajor> tet_;
    const geometry::Vector<4> weights_;
    geometry::Vector<4> values_;
    std::array<bool, 4> populated_{false, false, false, false};
  };

  struct Seed {
    LatticeCoordinates lc;
    geometry::Vector3 corrector;
    int k{};
  };

  struct Vertex {
    LatticeCoordinates node_lc;
    EdgeIndex ei{};
    Index vi{};
    double t0{};
    double v0{};
    double t1{};
    double v1{};
    double t2{};
    double v2{};

    geometry::Point3 position_clamped(const NodeList& node_list) const {
      const auto& node0 = node_list.at(node_lc);
      const auto& node1 = node_list.at(neighbor(node_lc, ei));
      const auto& p0 = node0.position();
      const auto& p1 = node1.position();
      auto t = std::clamp(t1, kVertexPositionMinimumOffset, 1.0 - kVertexPositionMinimumOffset);
      return p0 + t * (p1 - p0);
    }

    geometry::Point3 position_unclamped(const NodeList& node_list) const {
      const auto& node0 = node_list.at(node_lc);
      const auto& node1 = node_list.at(neighbor(node_lc, ei));
      const auto& p0 = node0.position();
      const auto& p1 = node1.position();
      auto t = t1;
      return p0 + t * (p1 - p0);
    }
  };

  // Returns true if the node is added.
  bool add_node(const LatticeCoordinates& lc) {
    if (node_list_.contains(lc)) {
      return false;
    }

    return add_node_unchecked(lc);
  }

  // Returns true if the node is added.
  bool add_node_unchecked(const LatticeCoordinates& lc) {
    auto p = position(lc);

    if (!second_extended_bbox().contains(p)) {
      return false;
    }

    node_list_.emplace(lc, Node(p));
    nodes_to_evaluate_.push_back(lc);

    return true;
  }

  static geometry::Vector<4> barycentric_coordinates(const geometry::Point3& p,
                                                     const geometry::Point3& a,
                                                     const geometry::Point3& b,
                                                     const geometry::Point3& c,
                                                     const geometry::Point3& d) {
    auto volume = [](const geometry::Point3& aa, const geometry::Point3& bb,
                     const geometry::Point3& cc, const geometry::Point3& dd) -> double {
      return (bb - aa).cross(cc - aa).dot(dd - aa) / 6.0;
    };

    auto v = volume(a, b, c, d);
    auto va = volume(b, d, c, p);
    auto vb = volume(a, c, d, p);
    auto vc = volume(a, d, b, p);
    auto vd = volume(a, b, c, p);
    return geometry::Vector<4>(va, vb, vc, vd) / v;
  }

  Index clustered_vertex_index(Index vi) const {
    return cluster_map_.contains(vi) ? cluster_map_.at(vi) : vi;
  }

  // Evaluates field values for each node in nodes_to_evaluate_.
  void evaluate_field(const FieldFunction& field_fn, double isovalue) {
    if (nodes_to_evaluate_.empty()) {
      return;
    }

    geometry::Points3 points(nodes_to_evaluate_.size(), 3);

    auto point_it = points.rowwise().begin();
    for (const auto& lc : nodes_to_evaluate_) {
      *point_it++ = node_list_.at(lc).position();
    }

    VecX values = field_fn(points).array() - isovalue;
    auto& arb_value = value_at_arbitrary_point_.value();

    Index i{};
    for (const auto& lc : nodes_to_evaluate_) {
      auto value = values(i);
      node_list_.at(lc).set_value(value);
      arb_value.tell(lc, value);
      i++;
    }

    nodes_to_evaluate_.clear();
  }

  void generate_vertices(const std::vector<LatticeCoordinates>& node_lcs) {
#pragma omp parallel for
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (std::size_t i = 0; i < node_lcs.size(); i++) {
      const auto& lc0 = node_lcs.at(i);
      auto& node0 = node_list_.at(lc0);
      auto v0 = node0.value();
      auto sign_v0 = node0.value_sign();

      for (EdgeIndex ei = 0; ei < 14; ei++) {
        auto lc1 = neighbor(lc0, ei);
        if (!LatticeCoordinatesLess()(lc1, lc0)) {
          continue;
        }

        auto* node1_ptr = node_list_.node_ptr(lc1);
        if (node1_ptr == nullptr) {
          // There is no neighbor node on the opposite end of the edge.
          continue;
        }

        auto& node1 = *node1_ptr;
        auto v1 = node1.value();
        auto sign_v1 = node1.value_sign();

        if (sign_v0 == sign_v1) {
          // There is no intersection on the edge.
          continue;
        }

        auto t = v0 / (v0 - v1);

#pragma omp critical
        {
          auto vi = static_cast<Index>(vertices_.size());
          auto opp_ei = kOppositeEdge.at(ei);

          if (t < 0.5) {
            node0.insert_vertex(vi, ei);
            vertices_.emplace_back(lc0, ei, vi,                                  //
                                   0.0, v0,                                      //
                                   t, std::numeric_limits<double>::quiet_NaN(),  //
                                   1.0, v1);
          } else {
            node1.insert_vertex(vi, opp_ei);
            vertices_.emplace_back(lc1, opp_ei, vi,                                    //
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

  geometry::Points3 get_vertices() const {
    geometry::Points3 vertices(static_cast<Index>(vertices_.size() + clustered_vertices_.size()),
                               3);
    auto it = vertices.rowwise().begin();
    for (const auto& v : vertices_) {
      *it++ = v.position_clamped(node_list_);
    }
    for (const auto& v : clustered_vertices_) {
      *it++ = v;
    }

    // To reduce the risk of generating near-degenerate faces during surface clipping,
    // snap vertices that are very close to the bbox .

    const auto& min = bbox().min();
    const auto& max = bbox().max();
    auto tiny = 1e-10 * resolution();

    for (auto p : vertices.rowwise()) {
      p = ((p.array() - min.array()).abs() < tiny).select(min, p);
      p = ((p.array() - max.array()).abs() < tiny).select(max, p);
    }

    return vertices;
  }

  geometry::Vector3 gradient(const LatticeCoordinates& lc) const {
    std::array<LatticeCoordinates, 7> lcs{lc,
                                          neighbor(lc, Edge::k1),
                                          neighbor(lc, Edge::k3),
                                          neighbor(lc, Edge::k5),
                                          neighbor(lc, Edge::k8),
                                          neighbor(lc, Edge::kA),
                                          neighbor(lc, Edge::kC)};
    Mat<7, 4> a;
    Vec<7> b;
    for (Index i = 0; i < 7; i++) {
      const auto& n = node_list_.at(lcs.at(i));
      a.row(i) << n.position(), 1.0;
      b(i) = std::abs(n.value());
    }
    Vec<4> x = (a.transpose() * a).ldlt().solve(a.transpose() * b);

    return x.head<3>().transpose();
  }

  bool is_boundary_node(const LatticeCoordinates& lc) const {
    for (EdgeIndex ei = 0; ei < 14; ei++) {
      auto p = position(neighbor(lc, ei));
      if (!second_extended_bbox().contains(p)) {
        return true;
      }
    }
    return false;
  }

  std::vector<LatticeCoordinates> knn_nodes(const LatticeCoordinates& lc, int k) const {
    std::unordered_set<LatticeCoordinates, LatticeCoordinatesHash> visited;
    visited.insert(lc);

    std::vector<LatticeCoordinates> frontier{lc};
    std::vector<LatticeCoordinates> next_frontier;

    for (auto i = 0; i < k; i++) {
      for (const auto& nlc : frontier) {
        for (EdgeIndex ei = 0; ei < 14; ei++) {
          auto nnlc = neighbor(nlc, ei);
          if (visited.contains(nnlc)) {
            continue;
          }
          auto p = position(nnlc);
          if (!second_extended_bbox().contains(p)) {
            continue;
          }
          visited.insert(nnlc);
          next_frontier.push_back(nnlc);
        }
      }

      std::swap(frontier, next_frontier);
      next_frontier.clear();
    }

    return frontier;
  }

  static void refine_vertices(std::vector<Vertex>& vertices) {
    using Mat = Mat3;
    using Vec = Vec<3>;

    for (auto& v : vertices) {
      // Solve y = a x^2 + b x + c for a, b, c with (x, y) = (t0, v0), (t1, v1), (t2, v2).
      Mat a;
      a << v.t0 * v.t0, v.t0, 1.0, v.t1 * v.t1, v.t1, 1.0, v.t2 * v.t2, v.t2, 1.0;
      Vec b(v.v0, v.v1, v.v2);
      Eigen::ColPivHouseholderQR<Mat> qr_a(a);
      if (!qr_a.isInvertible()) {
        continue;
      }
      Vec c = qr_a.solve(b);

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
  void remove_free_nodes(const std::vector<LatticeCoordinates>& node_lcs) {
    for (const auto& lc : node_lcs) {
      auto it = node_list_.find(lc);
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

  NodeList node_list_;
  std::vector<LatticeCoordinates> nodes_to_evaluate_;
  std::vector<Vertex> vertices_;
  std::unordered_map<Index, Index> cluster_map_;
  std::vector<geometry::Point3> clustered_vertices_;
  std::optional<InterpolatedValue> value_at_arbitrary_point_;
};

}  // namespace polatory::isosurface::rmt
