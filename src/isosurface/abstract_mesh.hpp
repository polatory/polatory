#pragma once

#include <Eigen/Core>
#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <iterator>
#include <jizai/isosurface/edge.hpp>
#include <jizai/isosurface/types.hpp>
#include <jizai/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jizai::isosurface {

// Side k of face fi is i = 4 * fi + k; the stride of 4 lets bit ops decode it.
struct Halfedge {
  Index i{-1};

  bool is_valid() const { return i >= 0; }

  bool operator==(const Halfedge&) const = default;
};

class VertexFaceRange {
 public:
  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Index;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = Index;

    Iterator() = default;

    explicit Iterator(const Halfedge* p) : p_(p) {}

    Index operator*() const { return p_->i >> 2; }

    Iterator& operator++() {
      ++p_;
      return *this;
    }

    Iterator operator++(int) {
      auto it = *this;
      ++p_;
      return it;
    }

    bool operator==(const Iterator&) const = default;

   private:
    const Halfedge* p_ = nullptr;
  };

  VertexFaceRange(const Halfedge* begin, const Halfedge* end) : begin_(begin), end_(end) {}

  Iterator begin() const { return Iterator{begin_}; }

  Iterator end() const { return Iterator{end_}; }

 private:
  const Halfedge* begin_;
  const Halfedge* end_;
};

class VertexOutgoingHalfedgeRange {
 public:
  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Halfedge;
    using difference_type = std::ptrdiff_t;
    using pointer = const Halfedge*;
    using reference = const Halfedge&;

    Iterator() = default;

    explicit Iterator(const Halfedge* p) : p_(p) {}

    reference operator*() const { return *p_; }

    Iterator& operator++() {
      ++p_;
      return *this;
    }

    Iterator operator++(int) {
      auto it = *this;
      ++p_;
      return it;
    }

    bool operator==(const Iterator&) const = default;

   private:
    const Halfedge* p_ = nullptr;
  };

  VertexOutgoingHalfedgeRange(const Halfedge* begin, const Halfedge* end)
      : begin_(begin), end_(end) {}

  Iterator begin() const { return Iterator{begin_}; }

  Iterator end() const { return Iterator{end_}; }

 private:
  const Halfedge* begin_;
  const Halfedge* end_;
};

// The connectivity of an oriented manifold triangle mesh. Face indices are stable.
class AbstractMesh {
 public:
  explicit AbstractMesh(Faces faces)
      : faces_(std::move(faces)), nf_(faces_.rows()), deleted_(nf_, false), opp_(4 * nf_) {
    for (Index fi = 0; fi < nf_; fi++) {
      register_face(fi);
    }
  }

  explicit AbstractMesh(Index capacity) : faces_(capacity, 3) { opp_.reserve(4 * capacity); }

  Index add_face(const Face& f) {
    auto fi = nf_++;
    faces_.row(fi) = f;
    deleted_.push_back(false);
    opp_.resize(4 * nf_);
    register_face(fi);
    return fi;
  }

  Index apex(Halfedge h) const { return h.is_valid() ? faces_(h.i >> 2, cw(h.i & 3)) : -1; }

  // Merges from(h) into to(h) and returns the retargeted faces. The result must be manifold.
  std::vector<Index> collapse(Halfedge h) {
    auto v_drop = from(h);
    auto v_keep = to(h);
    auto out = vertex_faces(v_drop);
    std::vector<Index> star(out.begin(), out.end());  // copy: retargeting rewrites the adjacency
    // Unregister all first: a retargeted face may take a side that another face in the star holds.
    for (auto fi : star) {
      unregister_face(fi);
    }
    std::vector<Index> moved;
    for (auto fi : star) {
      Face f = faces_.row(fi);
      if ((f.array() == v_keep).any()) {  // on the collapsed edge
        deleted_.at(fi) = true;
        continue;
      }
      faces_.row(fi) = (f.array() == v_drop).select(v_keep, f);
      register_face(fi);
      moved.push_back(fi);
    }
    return moved;
  }

  Face face(Index fi) const { return faces_.row(fi); }

  Index face(Halfedge h) const { return h.is_valid() ? h.i >> 2 : -1; }

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

  // e must be an interior edge.
  void flip(const Edge& e) {
    auto h0 = halfedge_of(e.a, e.b);
    auto h1 = halfedge_of(e.b, e.a);
    auto fi0 = face(h0);
    auto fi1 = face(h1);
    auto c = apex(h0);
    auto d = apex(h1);
    // Unregister both first: each new face takes a side that the other old face holds.
    unregister_face(fi0);
    unregister_face(fi1);
    faces_.row(fi0) = Face{c, e.a, d};
    faces_.row(fi1) = Face{d, e.b, c};
    register_face(fi0);
    register_face(fi1);
  }

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

  Index from(Halfedge h) const { return faces_(h.i >> 2, h.i & 3); }

  Halfedge halfedge(Index fi, int k) const { return {4 * fi + k}; }

  Halfedge halfedge_of(Index from, Index to) const {
    for (auto h : vertex_outgoing_halfedges(from)) {
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

  // v must be a new vertex.
  void insert_on_edge(const Edge& e, Index v) {
    auto sides = faces_of(e);
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

  Halfedge next(Halfedge h) const { return {(h.i & ~Index{3}) + ccw(h.i & 3)}; }

  Index num_faces() const { return nf_; }

  Halfedge opposite(Halfedge h) const { return h.is_valid() ? opp_.at(h.i) : Halfedge{}; }

  Halfedge prev(Halfedge h) const { return {(h.i & ~Index{3}) + cw(h.i & 3)}; }

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

  Index to(Halfedge h) const { return faces_(h.i >> 2, ccw(h.i & 3)); }

  VertexFaceRange vertex_faces(Index v) const {
    static const std::vector<Halfedge> none;
    const auto& hs = v < static_cast<Index>(outgoing_.size()) ? outgoing_.at(v) : none;
    return {hs.data(), hs.data() + hs.size()};
  }

  VertexOutgoingHalfedgeRange vertex_outgoing_halfedges(Index v) const {
    static const std::vector<Halfedge> none;
    const auto& hs = v < static_cast<Index>(outgoing_.size()) ? outgoing_.at(v) : none;
    return {hs.data(), hs.data() + hs.size()};
  }

 private:
  static Index ccw(Index k) { return k == 2 ? 0 : k + 1; }
  static Index cw(Index k) { return k == 0 ? 2 : k - 1; }

  void register_face(Index fi) {
    Face f = faces_.row(fi);
    // fi's halfedges are not in outgoing_ yet, so the lookups below cannot find fi itself.
    for (auto k = 0; k < 3; k++) {
      Halfedge h{4 * fi + k};
      Index a = f(k);
      Index b = f((k + 1) % 3);
      if (halfedge_of(a, b).is_valid()) {
        throw std::runtime_error("non-manifold or inconsistently oriented edge");
      }
      if (auto opp_h = halfedge_of(b, a); opp_h.is_valid()) {
        opp_.at(h.i) = opp_h;
        opp_.at(opp_h.i) = h;
      } else {
        opp_.at(h.i) = Halfedge{};
      }
    }
    for (auto k = 0; k < 3; k++) {
      auto v = f(k);
      if (v >= static_cast<Index>(outgoing_.size())) {
        outgoing_.resize(v + 1);
      }
      outgoing_.at(v).push_back(Halfedge{4 * fi + k});
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
        opp_.at(opp_h.i) = Halfedge{};
        opp_.at(h.i) = Halfedge{};
      }
      std::erase(outgoing_.at(f(k)), h);
    }
  }

  Faces faces_;
  Index nf_{};
  std::vector<bool> deleted_;
  std::vector<Halfedge> opp_;
  std::vector<std::vector<Halfedge>> outgoing_;
};

}  // namespace jizai::isosurface
