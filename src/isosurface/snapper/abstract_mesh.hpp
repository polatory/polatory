#pragma once

#include <Eigen/Core>
#include <array>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <unordered_map>

namespace polatory::isosurface::snapper {

// A triangle mesh's connectivity (faces are vertex-index triples, no coordinates). Each edge stores
// its two oriented sides -- the face traversing it as a -> b (a < b) and the one traversing it
// reversed, -1 if absent -- so the mesh must stay orientable and manifold: a second face on the same
// side throws. Faces have stable indices.
class AbstractMesh {
  using Sides = std::array<Index, 2>;

 public:
  explicit AbstractMesh(Faces faces) : faces_(std::move(faces)), nf_(faces_.rows()) {
    for (Index fi = 0; fi < nf_; fi++) {
      register_edges(fi);
    }
  }

  // Reserve capacity for an incremental build via add_face.
  explicit AbstractMesh(Index capacity) : faces_(capacity, 3) {}

  Index across(const Edge& e, Index fi) const {
    auto it = ef_.find(e);
    if (it == ef_.end()) {
      return -1;
    }
    auto [s0, s1] = it->second;
    if (s0 == fi) {
      return s1;
    }
    if (s1 == fi) {
      return s0;
    }
    return -1;
  }

  Index add_face(const Face& f) {
    auto fi = nf_++;
    faces_.row(fi) = f;
    register_edges(fi);
    return fi;
  }

  Face face(Index fi) const { return faces_.row(fi); }

  const Sides& faces_of(const Edge& e) const {
    static const Sides none{-1, -1};
    auto it = ef_.find(e);
    return it == ef_.end() ? none : it->second;
  }

  // Precondition: e has two faces (an interior edge).
  std::array<Index, 2> flip(const Edge& e) {
    auto [fi0, fi1] = faces_of(e);  // fi0 traverses e.a -> e.b, fi1 the reverse
    auto c = opposite(face(fi0), e);
    auto d = opposite(face(fi1), e);
    // Remove both old faces before adding either: a new face shares an outer edge with the other
    // old face in the same direction, so registering it while that face is still present would clash.
    unregister_edges(fi0);
    unregister_edges(fi1);
    faces_.row(fi0) = Face{c, e.a, d};
    faces_.row(fi1) = Face{d, e.b, c};
    register_edges(fi0);
    register_edges(fi1);
    return {fi0, fi1};
  }

  template <class Fn>
  void for_each_edge(const Fn& fn) const {
    for (const auto& [e, sides] : ef_) {
      fn(e, sides);
    }
  }

  bool has_edge(const Edge& e) const { return ef_.contains(e); }

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
      if (fi < 0) {
        continue;
      }
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

  Index num_faces() const { return nf_; }

  static Index opposite(const Face& f, const Edge& e) {
    for (auto v : f) {
      if (v != e.a && v != e.b) {
        return v;
      }
    }
    return -1;
  }

  Faces take_faces() && {
    faces_.conservativeResize(nf_, Eigen::NoChange);
    return std::move(faces_);
  }

 private:
  void register_edges(Index fi) {
    Face f = faces_.row(fi);
    for (auto k = 0; k < 3; k++) {
      Index u = f(k);
      Index w = f((k + 1) % 3);
      auto& sides = ef_.try_emplace(Edge{u, w}, Sides{-1, -1}).first->second;
      auto slot = u < w ? 0 : 1;
      if (sides.at(slot) >= 0) {
        throw std::runtime_error("non-manifold or inconsistently oriented edge");
      }
      sides.at(slot) = fi;
    }
  }

  void set_face(Index fi, const Face& f) {
    unregister_edges(fi);
    faces_.row(fi) = f;
    register_edges(fi);
  }

  void unregister_edges(Index fi) {
    Face f = faces_.row(fi);
    for (auto k = 0; k < 3; k++) {
      auto it = ef_.find({f(k), f((k + 1) % 3)});
      if (it == ef_.end()) {
        continue;
      }
      auto& sides = it->second;
      if (sides.at(0) == fi) {
        sides.at(0) = -1;
      }
      if (sides.at(1) == fi) {
        sides.at(1) = -1;
      }
      if (sides.at(0) < 0 && sides.at(1) < 0) {
        ef_.erase(it);
      }
    }
  }

  Faces faces_;
  Index nf_{};
  std::unordered_map<Edge, Sides, EdgeHash> ef_;
};

}  // namespace polatory::isosurface::snapper
