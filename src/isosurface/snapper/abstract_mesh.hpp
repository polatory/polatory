#pragma once

#include <Eigen/Core>
#include <array>
#include <boost/container/static_vector.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::isosurface::snapper {

// A triangle mesh's connectivity (faces are vertex-index triples, no coordinates). Each directed
// edge (halfedge) maps to its incident face, so the two halfedges of an edge are its two oriented
// sides; a second face on the same halfedge throws, keeping the mesh orientable and manifold. Faces
// have stable indices.
class AbstractMesh {
  using Halfedge = std::pair<Index, Index>;

  struct HalfedgeHash {
    std::size_t operator()(const Halfedge& he) const noexcept {
      std::size_t seed{};
      boost::hash_combine(seed, he.first);
      boost::hash_combine(seed, he.second);
      return seed;
    }
  };

 public:
  explicit AbstractMesh(Faces faces)
      : faces_(std::move(faces)), nf_(faces_.rows()), deleted_(nf_, false) {
    for (Index fi = 0; fi < nf_; fi++) {
      register_edges(fi);
    }
  }

  // Reserve capacity for an incremental build via add_face.
  explicit AbstractMesh(Index capacity) : faces_(capacity, 3) {}

  Index across(const Edge& e, Index fi) const {
    auto f_ab = face_on(e.a, e.b);
    auto f_ba = face_on(e.b, e.a);
    if (f_ab == fi) {
      return f_ba;
    }
    if (f_ba == fi) {
      return f_ab;
    }
    return -1;
  }

  Index add_face(const Face& f) {
    auto fi = nf_++;
    faces_.row(fi) = f;
    deleted_.push_back(false);
    register_edges(fi);
    return fi;
  }

  // Collapses edge e onto its endpoint v_keep: the faces on e are deleted, and the rest of the
  // dropped endpoint's star is retargeted to v_keep. Precondition: the result is manifold (the
  // caller checks the link condition). Returns the retargeted faces.
  std::vector<Index> collapse(const Edge& e, Index v_keep) {
    auto v_drop = e.a == v_keep ? e.b : e.a;
    auto star = incident(v_drop);  // copy: retargeting rewrites the adjacency
    // Remove every face before retargeting any: a retargeted face claims an edge side vacated by
    // a deleted face, so registering it while that face is still present would clash.
    for (auto fi : star) {
      unregister_edges(fi);
    }
    std::vector<Index> moved;
    for (auto fi : star) {
      Face f = faces_.row(fi);
      if ((f.array() == v_keep).any()) {
        deleted_.at(fi) = true;  // a face on e collapses to a degenerate sliver
        continue;
      }
      faces_.row(fi) = (f.array() == v_drop).select(v_keep, f);
      register_edges(fi);
      moved.push_back(fi);
    }
    return moved;
  }

  Face face(Index fi) const { return faces_.row(fi); }

  // The faces (at most two) incident to e, the a -> b side first when both are present.
  boost::container::static_vector<Index, 2> faces_of(const Edge& e) const {
    boost::container::static_vector<Index, 2> fs;
    if (auto fi = face_on(e.a, e.b); fi >= 0) {
      fs.push_back(fi);
    }
    if (auto fi = face_on(e.b, e.a); fi >= 0) {
      fs.push_back(fi);
    }
    return fs;
  }

  // Precondition: e has two faces (an interior edge).
  std::array<Index, 2> flip(const Edge& e) {
    auto sides = faces_of(e);  // interior edge: sides[0] traverses e.a -> e.b, sides[1] the reverse
    auto fi0 = sides[0];
    auto fi1 = sides[1];
    auto c = opposite(face(fi0), e);
    auto d = opposite(face(fi1), e);
    // Remove both old faces before adding either: a new face shares an outer edge with the other
    // old face in the same direction, so registering it while that face is still present would
    // clash.
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
    for (const auto& [he, fi] : he_) {
      auto [u, w] = he;
      // Visit each undirected edge once: from its a -> b halfedge if present, else the lone b -> a.
      if (u < w) {
        fn(Edge{u, w}, std::array<Index, 2>{fi, face_on(w, u)});
      } else if (face_on(w, u) < 0) {
        fn(Edge{u, w}, std::array<Index, 2>{-1, fi});
      }
    }
  }

  bool has_edge(const Edge& e) const {
    return he_.contains({e.a, e.b}) || he_.contains({e.b, e.a});
  }

  const std::vector<Index>& incident(Index v) const {
    static const std::vector<Index> none;
    return v < static_cast<Index>(vf_.size()) ? vf_.at(v) : none;
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
  // The face traversing the directed edge u -> w, or -1 if none.
  Index face_on(Index u, Index w) const {
    auto it = he_.find({u, w});
    return it != he_.end() ? it->second : -1;
  }

  void register_edges(Index fi) {
    Face f = faces_.row(fi);
    for (auto k = 0; k < 3; k++) {
      if (!he_.emplace(Halfedge{f(k), f((k + 1) % 3)}, fi).second) {
        throw std::runtime_error("non-manifold or inconsistently oriented edge");
      }
    }
    for (auto v : f) {
      if (v >= static_cast<Index>(vf_.size())) {
        vf_.resize(v + 1);
      }
      vf_.at(v).push_back(fi);
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
      he_.erase({f(k), f((k + 1) % 3)});
    }
    for (auto v : f) {
      std::erase(vf_.at(v), fi);
    }
  }

  Faces faces_;
  Index nf_{};
  std::vector<bool> deleted_;
  boost::unordered_flat_map<Halfedge, Index, HalfedgeHash> he_;
  std::vector<std::vector<Index>> vf_;
};

}  // namespace polatory::isosurface::snapper
