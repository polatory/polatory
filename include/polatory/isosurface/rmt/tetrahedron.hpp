#pragma once

#include <array>
#include <boost/iterator/iterator_facade.hpp>
#include <iterator>
#include <optional>
#include <polatory/common/macros.hpp>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/rmt/neighbor.hpp>
#include <polatory/isosurface/rmt/node.hpp>
#include <polatory/isosurface/rmt/node_list.hpp>
#include <polatory/isosurface/sign.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface::rmt {

class Tetrahedron {
  // List of indices of the three edges of each tetrahedron.
  static constexpr std::array<std::array<EdgeIndex, 3>, 6> kEdgeIndices{
      {{Edge::k4, Edge::k5, Edge::k2},
       {Edge::k4, Edge::k2, Edge::k1},
       {Edge::k4, Edge::k1, Edge::k0},
       {Edge::k4, Edge::k0, Edge::k3},
       {Edge::k4, Edge::k3, Edge::k6},
       {Edge::k4, Edge::k6, Edge::k5}}};

  // List of indices of the three outer edges of each tetrahedron as:
  //          ei0           ei1           ei2
  //  node0 ------> node1 ------> node2 ------> node0
  //        <------       <------       <------
  //         ~ei0          ~ei1          ~ei2
  // where
  //  ei0: outer edge from node0 to node1
  //  ~ei0: opposite outer edge of e0 from node1 to node0.
  static constexpr std::array<std::array<EdgeIndex, 3>, 6> kOuterEdgeIndices{
      {{Edge::k7, Edge::kD, Edge::k3},
       {Edge::kA, Edge::k0, Edge::k6},
       {Edge::kD, Edge::k9, Edge::k5},
       {Edge::kC, Edge::k6, Edge::k2},
       {Edge::k9, Edge::k7, Edge::k1},
       {Edge::k8, Edge::k2, Edge::k0}}};

 public:
  Tetrahedron(const LatticeCoordinates& lc, const NodeList& node_list, int index)
      : lc_(lc), node_list_(node_list), index_(index) {}

  // Adds 0, 1 or 2 triangular faces of the isosurface in this tetrahedron.
  template <class OutputIterator>
  void get_faces(OutputIterator faces) const {
    auto ei0 = kEdgeIndices.at(index_)[0];
    auto ei1 = kEdgeIndices.at(index_)[1];
    auto ei2 = kEdgeIndices.at(index_)[2];

    const auto& n = node_list_.at(lc_);
    const auto& n0 = node_list_.at(neighbor(lc_, ei0));
    const auto& n1 = node_list_.at(neighbor(lc_, ei1));
    const auto& n2 = node_list_.at(neighbor(lc_, ei2));

    auto oei0 = kOuterEdgeIndices.at(index_)[0];
    auto oei1 = kOuterEdgeIndices.at(index_)[1];
    auto oei2 = kOuterEdgeIndices.at(index_)[2];

    // Possible vertices on the six edges of the tetrahedron.
    auto v0 = vertex_on_edge(n, ei0, n0);
    auto v1 = vertex_on_edge(n, ei1, n1);
    auto v2 = vertex_on_edge(n, ei2, n2);
    auto v3 = vertex_on_edge(n0, oei0, n1);
    auto v4 = vertex_on_edge(n1, oei1, n2);
    auto v5 = vertex_on_edge(n2, oei2, n0);

    auto make_class = [](BinarySign a, BinarySign b, BinarySign c, BinarySign d) constexpr -> int {
      return (a == BinarySign::kNeg ? 8 : 0) + (b == BinarySign::kNeg ? 4 : 0) +
             (c == BinarySign::kNeg ? 2 : 0) + (d == BinarySign::kNeg ? 1 : 0);
    };
    switch (make_class(n.value_sign(), n0.value_sign(), n1.value_sign(), n2.value_sign())) {
      case make_class(BinarySign::kPos, BinarySign::kPos, BinarySign::kPos, BinarySign::kPos):
      case make_class(BinarySign::kNeg, BinarySign::kNeg, BinarySign::kNeg, BinarySign::kNeg):
        // No faces.
        break;
      case make_class(BinarySign::kNeg, BinarySign::kPos, BinarySign::kPos, BinarySign::kPos):
        *faces++ = {v0.value(), v1.value(), v2.value()};
        break;
      case make_class(BinarySign::kPos, BinarySign::kNeg, BinarySign::kNeg, BinarySign::kNeg):
        *faces++ = {v0.value(), v2.value(), v1.value()};
        break;
      case make_class(BinarySign::kPos, BinarySign::kNeg, BinarySign::kPos, BinarySign::kPos):
        *faces++ = {v0.value(), v5.value(), v3.value()};
        break;
      case make_class(BinarySign::kNeg, BinarySign::kPos, BinarySign::kNeg, BinarySign::kNeg):
        *faces++ = {v0.value(), v3.value(), v5.value()};
        break;
      case make_class(BinarySign::kPos, BinarySign::kPos, BinarySign::kNeg, BinarySign::kPos):
        *faces++ = {v1.value(), v3.value(), v4.value()};
        break;
      case make_class(BinarySign::kNeg, BinarySign::kNeg, BinarySign::kPos, BinarySign::kNeg):
        *faces++ = {v1.value(), v4.value(), v3.value()};
        break;
      case make_class(BinarySign::kPos, BinarySign::kPos, BinarySign::kPos, BinarySign::kNeg):
        *faces++ = {v2.value(), v4.value(), v5.value()};
        break;
      case make_class(BinarySign::kNeg, BinarySign::kNeg, BinarySign::kNeg, BinarySign::kPos):
        *faces++ = {v2.value(), v5.value(), v4.value()};
        break;
      case make_class(BinarySign::kNeg, BinarySign::kNeg, BinarySign::kPos, BinarySign::kPos):
        *faces++ = {v5.value(), v3.value(), v1.value()};
        *faces++ = {v5.value(), v1.value(), v2.value()};
        break;
      case make_class(BinarySign::kPos, BinarySign::kPos, BinarySign::kNeg, BinarySign::kNeg):
        *faces++ = {v5.value(), v1.value(), v3.value()};
        *faces++ = {v5.value(), v2.value(), v1.value()};
        break;
      case make_class(BinarySign::kNeg, BinarySign::kPos, BinarySign::kNeg, BinarySign::kPos):
        *faces++ = {v0.value(), v3.value(), v4.value()};
        *faces++ = {v0.value(), v4.value(), v2.value()};
        break;
      case make_class(BinarySign::kPos, BinarySign::kNeg, BinarySign::kPos, BinarySign::kNeg):
        *faces++ = {v0.value(), v4.value(), v3.value()};
        *faces++ = {v0.value(), v2.value(), v4.value()};
        break;
      case make_class(BinarySign::kNeg, BinarySign::kPos, BinarySign::kPos, BinarySign::kNeg):
        *faces++ = {v5.value(), v0.value(), v1.value()};
        *faces++ = {v5.value(), v1.value(), v4.value()};
        break;
      case make_class(BinarySign::kPos, BinarySign::kNeg, BinarySign::kNeg, BinarySign::kPos):
        *faces++ = {v5.value(), v1.value(), v0.value()};
        *faces++ = {v5.value(), v4.value(), v1.value()};
        break;
      default:
        POLATORY_UNREACHABLE();
        break;
    }
  }

 private:
  friend class TetrahedronIterator;

  static std::optional<Index> vertex_on_edge(const Node& node, EdgeIndex edge_idx,
                                             const Node& opp_node) {
    if (node.has_vertex(edge_idx)) {
      return node.vertex(edge_idx);
    }

    auto opp_edge_idx = kOppositeEdge.at(edge_idx);
    if (opp_node.has_vertex(opp_edge_idx)) {
      return opp_node.vertex(opp_edge_idx);
    }

    return {};
  }

  const LatticeCoordinates& lc_;
  const NodeList& node_list_;
  const int index_;
};

class TetrahedronIterator
    : public boost::iterator_facade<TetrahedronIterator, Tetrahedron, std::input_iterator_tag,
                                    Tetrahedron, int> {
  // The number of tetrahedra in a cell.
  static constexpr int kNumTetrahedra = 6;

 public:
  explicit TetrahedronIterator(const LatticeCoordinates& lc, const NodeList& node_list)
      : lc_(lc), node_list_(node_list) {
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

    return {lc_, node_list_, index_};
  }

  bool equal(const TetrahedronIterator& other) const {
    POLATORY_ASSERT(lc_ == other.lc_);
    POLATORY_ASSERT(std::addressof(node_list_) == std::addressof(other.node_list_));

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
    return node_list_.contains(neighbor(lc_, Tetrahedron::kEdgeIndices.at(index_)[0])) &&
           node_list_.contains(neighbor(lc_, Tetrahedron::kEdgeIndices.at(index_)[1])) &&
           node_list_.contains(neighbor(lc_, Tetrahedron::kEdgeIndices.at(index_)[2]));
  }

  const LatticeCoordinates& lc_;
  const NodeList& node_list_;
  int index_{};
};

}  // namespace polatory::isosurface::rmt
