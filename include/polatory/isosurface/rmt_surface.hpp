#pragma once

#include <array>
#include <boost/iterator/iterator_facade.hpp>
#include <iterator>
#include <memory>
#include <optional>
#include <polatory/common/macros.hpp>
#include <polatory/isosurface/rmt_lattice.hpp>
#include <polatory/isosurface/rmt_node.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

namespace detail {

class rmt_tetrahedron {
  // List of indices of the three edges of each tetrahedron.
  static constexpr std::array<std::array<edge_index, 3>, 6> EdgeIndices{
      {{0, 1, 4}, {0, 4, 3}, {0, 3, 9}, {0, 9, 12}, {0, 12, 13}, {0, 13, 1}}};

  // List of indices of the three outer edges of each tetrahedron as:
  //          ei0           ei1           ei2
  //  node0 ------> node1 ------> node2 ------> node0
  //        <------       <------       <------
  //         ~ei0          ~ei1          ~ei2
  // where
  //  ei0: outer edge from node0 to node1
  //  ~ei0: opposite outer edge of e0 from node1 to node0.
  static constexpr std::array<std::array<edge_index, 3>, 6> OuterEdgeIndices{
      {{2, 6, 12}, {5, 9, 13}, {6, 11, 1}, {8, 13, 4}, {11, 2, 3}, {10, 4, 9}}};

  // Encode four signs of tetrahedron nodes into an integer.
  template <binary_sign s, binary_sign s0, binary_sign s1, binary_sign s2>
  static constexpr int tetrahedron_type() {
    return (s2 << 3) | (s1 << 2) | (s0 << 1) | s;
  }

 public:
  rmt_tetrahedron(const rmt_node& node, int index) : node_(node), index_(index) {}

  // Adds 0, 1 or 2 triangular faces of the isosurface in this tetrahedron.
  template <class OutputIterator>
  void get_faces(OutputIterator faces) const {
    auto ei0 = EdgeIndices.at(index_)[0];
    auto ei1 = EdgeIndices.at(index_)[1];
    auto ei2 = EdgeIndices.at(index_)[2];

    const auto& node0 = node_.neighbor(ei0);
    const auto& node1 = node_.neighbor(ei1);
    const auto& node2 = node_.neighbor(ei2);

    if (degenerate(node_, node0, node1, node2)) {
      return;
    }

    auto oei0 = OuterEdgeIndices.at(index_)[0];
    auto oei1 = OuterEdgeIndices.at(index_)[1];
    auto oei2 = OuterEdgeIndices.at(index_)[2];

    // Check six edges to obtain vertices

    auto v0 = vertex_on_edge(node_, ei0, node0);
    auto v1 = vertex_on_edge(node_, ei1, node1);
    auto v2 = vertex_on_edge(node_, ei2, node2);
    auto v3 = vertex_on_edge(node0, oei0, node1);
    auto v4 = vertex_on_edge(node1, oei1, node2);
    auto v5 = vertex_on_edge(node2, oei2, node0);

    auto tetra = tetrahedron_type(node_.value_sign(), node0.value_sign(), node1.value_sign(),
                                  node2.value_sign());

    switch (tetra) {
      case tetrahedron_type<Pos, Pos, Pos, Pos>():
      case tetrahedron_type<Neg, Neg, Neg, Neg>():
        // No faces.
        break;
      case tetrahedron_type<Neg, Pos, Pos, Pos>():
        // v0-v1-v2
        *faces++ = {v0.value(), v1.value(), v2.value()};
        break;
      case tetrahedron_type<Pos, Neg, Neg, Neg>():
        // v0-v2-v1
        *faces++ = {v0.value(), v2.value(), v1.value()};
        break;
      case tetrahedron_type<Pos, Neg, Pos, Pos>():
        // v0-v5-v3
        *faces++ = {v0.value(), v5.value(), v3.value()};
        break;
      case tetrahedron_type<Neg, Pos, Neg, Neg>():
        // v0-v3-v5
        *faces++ = {v0.value(), v3.value(), v5.value()};
        break;
      case tetrahedron_type<Pos, Pos, Neg, Pos>():
        // v1-v3-v4
        *faces++ = {v1.value(), v3.value(), v4.value()};
        break;
      case tetrahedron_type<Neg, Neg, Pos, Neg>():
        // v1-v4-v3
        *faces++ = {v1.value(), v4.value(), v3.value()};
        break;
      case tetrahedron_type<Pos, Pos, Pos, Neg>():
        // v2-v4-v5
        *faces++ = {v2.value(), v4.value(), v5.value()};
        break;
      case tetrahedron_type<Neg, Neg, Neg, Pos>():
        // v2-v5-v4
        *faces++ = {v2.value(), v5.value(), v4.value()};
        break;
      case tetrahedron_type<Neg, Neg, Pos, Pos>():
        // v5-v3-v1, v5-v1-v2
        *faces++ = {v5.value(), v3.value(), v1.value()};
        *faces++ = {v5.value(), v1.value(), v2.value()};
        break;
      case tetrahedron_type<Pos, Pos, Neg, Neg>():
        // v5-v1-v3, v5-v2-v1
        *faces++ = {v5.value(), v1.value(), v3.value()};
        *faces++ = {v5.value(), v2.value(), v1.value()};
        break;
      case tetrahedron_type<Neg, Pos, Neg, Pos>():
        // v0-v3-v4, v0-v4-v2
        *faces++ = {v0.value(), v3.value(), v4.value()};
        *faces++ = {v0.value(), v4.value(), v2.value()};
        break;
      case tetrahedron_type<Pos, Neg, Pos, Neg>():
        // v0-v4-v3, v0-v2-v4
        *faces++ = {v0.value(), v4.value(), v3.value()};
        *faces++ = {v0.value(), v2.value(), v4.value()};
        break;
      case tetrahedron_type<Neg, Pos, Pos, Neg>():
        // v5-v0-v1, v5-v1-v4
        *faces++ = {v5.value(), v0.value(), v1.value()};
        *faces++ = {v5.value(), v1.value(), v4.value()};
        break;
      case tetrahedron_type<Pos, Neg, Neg, Pos>():
        // v5-v1-v0, v5-v4-v1
        *faces++ = {v5.value(), v1.value(), v0.value()};
        *faces++ = {v5.value(), v4.value(), v1.value()};
        break;
      default:
        POLATORY_UNREACHABLE();
        break;
    }
  }

 private:
  friend class rmt_tetrahedron_iterator;

  // Test if the tetrahedron is degenerate due to clamping within the bbox.
  static bool degenerate(const rmt_node& node, const rmt_node& node0, const rmt_node& node1,
                         const rmt_node& node2) {
    const auto& p = node.position();
    const auto& p0 = node0.position();
    const auto& p1 = node1.position();
    const auto& p2 = node2.position();

    return (p.array() == p0.array() && p.array() == p1.array() && p.array() == p2.array()).any();
  }

  static int tetrahedron_type(binary_sign s, binary_sign s0, binary_sign s1, binary_sign s2) {
    return (s2 << 3) | (s1 << 2) | (s0 << 1) | s;
  }

  static std::optional<vertex_index> vertex_on_edge(const rmt_node& node, edge_index edge_idx,
                                                    const rmt_node& opp_node) {
    if (node.has_intersection(edge_idx)) {
      return node.vertex_on_edge(edge_idx);
    }

    auto opp_edge_idx = OppositeEdge.at(edge_idx);
    if (opp_node.has_intersection(opp_edge_idx)) {
      return opp_node.vertex_on_edge(opp_edge_idx);
    }

    return {};
  }

  const rmt_node& node_;
  const int index_;
};

class rmt_tetrahedron_iterator
    : public boost::iterator_facade<rmt_tetrahedron_iterator, rmt_tetrahedron,
                                    std::input_iterator_tag, rmt_tetrahedron, int> {
  // The number of tetrahedra in a cell.
  static constexpr int kNumTetrahedra = 6;

 public:
  explicit rmt_tetrahedron_iterator(const rmt_node& node) : node_(node) {
    while (is_valid() && !tetrahedron_exists()) {
      // Some of the tetrahedron nodes do not exist.
      index_++;
    }
  }

  bool is_valid() const { return index_ < kNumTetrahedra; }

 private:
  friend class boost::iterator_core_access;

  reference dereference() const {
    POLATORY_ASSERT(is_valid());

    return {node_, index_};
  }

  bool equal(const rmt_tetrahedron_iterator& other) const {
    POLATORY_ASSERT(std::addressof(node_) == std::addressof(other.node_));

    return index_ == other.index_;
  }

  void increment() {
    POLATORY_ASSERT(is_valid());

    do {
      index_++;
    } while (is_valid() && !tetrahedron_exists());
  }

  // Returns if all nodes corresponding to three vertices of the tetrahedron exist.
  bool tetrahedron_exists() const {
    return node_.has_neighbor(rmt_tetrahedron::EdgeIndices.at(index_)[0]) &&
           node_.has_neighbor(rmt_tetrahedron::EdgeIndices.at(index_)[1]) &&
           node_.has_neighbor(rmt_tetrahedron::EdgeIndices.at(index_)[2]);
  }

  const rmt_node& node_;
  int index_{};
};

}  // namespace detail

class rmt_surface {
 public:
  explicit rmt_surface(const rmt_lattice& lattice) : lattice_(lattice) {}

  Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor> get_faces() const {
    Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor> faces(
        static_cast<index_t>(faces_.size()), 3);

    index_t n_faces = 0;

    auto it = faces.rowwise().begin();
    for (const auto& face : faces_) {
      auto v0 = lattice_.clustered_vertex_index(face[0]);
      auto v1 = lattice_.clustered_vertex_index(face[1]);
      auto v2 = lattice_.clustered_vertex_index(face[2]);

      if (v0 == v1 || v1 == v2 || v2 == v0) {
        // Degenerated face.
        continue;
      }

      *it++ << v0, v1, v2;
      n_faces++;
    }

    faces.conservativeResize(n_faces, 3);
    return faces;
  }

  void generate_surface() {
    faces_.clear();

    auto inserter = std::back_inserter(faces_);

    for (const auto& ci_node : lattice_.node_list_) {
      const auto& node = ci_node.second;

      for (detail::rmt_tetrahedron_iterator it(node); it.is_valid(); ++it) {
        it->get_faces(inserter);
      }
    }
  }

 private:
  const rmt_lattice& lattice_;
  std::vector<face> faces_;
};

}  // namespace polatory::isosurface
