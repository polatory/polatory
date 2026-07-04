#pragma once

#include <Eigen/Core>
#include <boost/container/static_vector.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::isosurface::snapper {

// A directed side of a face, identified by its index 4 * fi + k (fi the face, k in 0..2) in the
// implicit halfedge list. The stride is 4, not 3, so fi = h.i >> 2 and k = h.i & 3 are bit ops
// rather than division by three; the fourth slot per face is unused.
struct Halfedge {
  Index i{-1};  // -1 for none

  bool is_valid() const { return i >= 0; }

  bool operator==(const Halfedge&) const = default;
};

// A triangle mesh's connectivity (faces are vertex-index triples, no coordinates). A second face on
// the same directed edge throws, keeping the mesh orientable and manifold. Faces have stable
// indices.
class AbstractMesh {
 public:
  explicit AbstractMesh(Faces faces)
      : faces_(std::move(faces)), nf_(faces_.rows()), deleted_(nf_, false), opp_(4 * nf_) {
    for (Index fi = 0; fi < nf_; fi++) {
      register_face(fi);
    }
  }

  // Reserve capacity for an incremental build via add_face.
  explicit AbstractMesh(Index capacity) : faces_(capacity, 3) { opp_.reserve(4 * capacity); }

  Index add_face(const Face& f) {
    auto fi = nf_++;
    faces_.row(fi) = f;
    deleted_.push_back(false);
    opp_.resize(4 * nf_);
    register_face(fi);
    return fi;
  }

  // The vertex of h's face opposite its edge (the h.to -> apex -> h.from fan), or -1 if h has no
  // face.
  Index apex(Halfedge h) const { return h.is_valid() ? faces_(h.i >> 2, cw(h.i & 3)) : -1; }

  // Collapses halfedge h, merging from(h) into to(h): the faces on the edge are deleted, and the
  // rest of from(h)'s star is retargeted to to(h). Precondition: the result is manifold (the caller
  // checks the link condition). Returns the retargeted faces.
  std::vector<Index> collapse(Halfedge h) {
    auto v_drop = from(h);
    auto v_keep = to(h);
    auto hs = outgoing(v_drop);  // copy: retargeting rewrites the adjacency
    // Remove every face before retargeting any: a retargeted face claims an edge side vacated by
    // a deleted face, so registering it while that face is still present would clash.
    for (auto hh : hs) {
      unregister_face(face(hh));
    }
    std::vector<Index> moved;
    for (auto hh : hs) {
      auto fi = face(hh);
      Face f = faces_.row(fi);
      if ((f.array() == v_keep).any()) {
        deleted_.at(fi) = true;  // a face on the collapsed edge becomes a degenerate sliver
        continue;
      }
      faces_.row(fi) = (f.array() == v_drop).select(v_keep, f);
      register_face(fi);
      moved.push_back(fi);
    }
    return moved;
  }

  Face face(Index fi) const { return faces_.row(fi); }

  // The face incident to halfedge h, or -1 if none (h is a boundary side).
  Index face(Halfedge h) const { return h.is_valid() ? h.i >> 2 : -1; }

  // The faces (at most two) incident to e, the a -> b side first when both are present.
  boost::container::static_vector<Index, 2> faces_of(const Edge& e) const {
    boost::container::static_vector<Index, 2> fs;
    if (auto fi = face(halfedge_of(e.a, e.b)); fi >= 0) {
      fs.push_back(fi);
    }
    if (auto fi = face(halfedge_of(e.b, e.a)); fi >= 0) {
      fs.push_back(fi);
    }
    return fs;
  }

  // Precondition: e has two faces (an interior edge).
  void flip(const Edge& e) {
    auto h0 = halfedge_of(e.a, e.b);  // traverses e.a -> e.b
    auto h1 = halfedge_of(e.b, e.a);  // the reverse side
    auto fi0 = face(h0);
    auto fi1 = face(h1);
    auto c = apex(h0);  // fi0's apex
    auto d = apex(h1);  // fi1's apex
    // Remove both old faces before adding either: a new face shares an outer edge with the other
    // old face in the same direction, so registering it while that face is still present would
    // clash.
    unregister_face(fi0);
    unregister_face(fi1);
    faces_.row(fi0) = Face{c, e.a, d};
    faces_.row(fi1) = Face{d, e.b, c};
    register_face(fi0);
    register_face(fi1);
  }

  // Visits each present halfedge (each directed side that has a face) once, in face-corner order.
  template <class Fn>
  void for_each_halfedge(const Fn& fn) const {
    for (Index fi = 0; fi < nf_; fi++) {
      if (deleted_.at(fi)) {
        continue;
      }
      for (auto k = 0; k < 3; k++) {
        fn(Halfedge{4 * fi + k});
      }
    }
  }

  // The tail vertex of h (h traverses from -> to).
  Index from(Halfedge h) const { return faces_(h.i >> 2, h.i & 3); }

  // Halfedge k of face fi: from vertex k to vertex k + 1.
  Halfedge halfedge(Index fi, int k) const { return {4 * fi + k}; }

  // The halfedge traversing from -> to, or an invalid halfedge if there is none. Found by scanning
  // from's outgoing halfedges.
  Halfedge halfedge_of(Index from, Index to) const {
    for (auto h : outgoing(from)) {
      if (this->to(h) == to) {
        return h;
      }
    }
    return {};
  }

  bool has_edge(const Edge& e) const {
    return halfedge_of(e.a, e.b).is_valid() || halfedge_of(e.b, e.a).is_valid();
  }

  // v must be a new vertex.
  void insert_in_face(Index fi, Index v) {
    auto f = face(fi);
    set_face(fi, {f(0), f(1), v});
    add_face({f(1), f(2), v});
    add_face({f(2), f(0), v});
  }

  // v must be new -- so, unlike flip, the split faces clash with no existing edge (no two-phase).
  void insert_on_edge(const Edge& e, Index v) {
    auto sides = faces_of(e);  // copy: set_face below rewrites the incidence
    for (auto fi : sides) {
      auto f = face(fi);
      auto i = 0;
      for (auto k = 0; k < 3; k++) {
        if ((f(k) == e.a && f((k + 1) % 3) == e.b) || (f(k) == e.b && f((k + 1) % 3) == e.a)) {
          i = k;
          break;
        }
      }
      set_face(fi, {f(i), v, f((i + 2) % 3)});
      add_face({v, f((i + 1) % 3), f((i + 2) % 3)});
    }
  }

  // The next halfedge around h's face.
  Halfedge next(Halfedge h) const { return {(h.i & ~Index{3}) + ccw(h.i & 3)}; }

  Index num_faces() const { return nf_; }

  // The halfedge on the other side of h's edge, or an invalid halfedge if h is a boundary side or
  // itself invalid.
  Halfedge opposite(Halfedge h) const { return h.is_valid() ? opp_.at(h.i) : Halfedge{}; }

  // v's outgoing halfedges, one per incident face (face(h) recovers the face).
  const std::vector<Halfedge>& outgoing(Index v) const {
    static const std::vector<Halfedge> none;
    return v < static_cast<Index>(vh_.size()) ? vh_.at(v) : none;
  }

  // The previous halfedge around h's face.
  Halfedge prev(Halfedge h) const { return {(h.i & ~Index{3}) + cw(h.i & 3)}; }

  // The head vertex of h.
  Index to(Halfedge h) const { return faces_(h.i >> 2, ccw(h.i & 3)); }

  Faces take_faces() && {
    Index n = 0;
    for (Index fi = 0; fi < nf_; fi++) {
      if (!deleted_.at(fi)) {
        faces_.row(n++) = faces_.row(fi);
      }
    }
    faces_.conservativeResize(n, Eigen::NoChange);
    return std::move(faces_);
  }

 private:
  // Rotate the local index k (= h.i & 3, in 0..2) forward/back within its triangle.
  static Index ccw(Index k) { return k == 2 ? 0 : k + 1; }
  static Index cw(Index k) { return k == 0 ? 2 : k - 1; }

  void register_face(Index fi) {
    Face f = faces_.row(fi);
    // fi is not yet in vf_, so the lookups below never find fi itself.
    for (auto k = 0; k < 3; k++) {
      Halfedge h{4 * fi + k};
      Index a = f(k);
      Index b = f((k + 1) % 3);
      if (halfedge_of(a, b).is_valid()) {
        throw std::runtime_error("non-manifold or inconsistently oriented edge");
      }
      if (auto opp_h = halfedge_of(b, a); opp_h.is_valid()) {  // pair the opposite
        opp_.at(h.i) = opp_h;
        opp_.at(opp_h.i) = h;
      } else {
        opp_.at(h.i) = Halfedge{};
      }
    }
    for (auto k = 0; k < 3; k++) {
      auto v = f(k);
      if (v >= static_cast<Index>(vh_.size())) {
        vh_.resize(v + 1);
      }
      vh_.at(v).push_back(Halfedge{4 * fi + k});
    }
  }

  void set_face(Index fi, const Face& f) {
    unregister_face(fi);
    faces_.row(fi) = f;
    register_face(fi);
  }

  void unregister_face(Index fi) {
    Face f = faces_.row(fi);
    for (auto k = 0; k < 3; k++) {
      Halfedge h{4 * fi + k};
      if (auto opp_h = opp_.at(h.i); opp_h.is_valid()) {
        opp_.at(opp_h.i) = Halfedge{};  // the opposite becomes a boundary side
        opp_.at(h.i) = Halfedge{};
      }
      std::erase(vh_.at(f(k)), h);
    }
  }

  Faces faces_;
  Index nf_{};
  std::vector<bool> deleted_;
  std::vector<Halfedge> opp_;  // opp_[4 * fi + k] = the opposite halfedge, invalid on the boundary
  std::vector<std::vector<Halfedge>> vh_;  // vertex -> its outgoing halfedges
};

}  // namespace polatory::isosurface::snapper
