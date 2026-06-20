#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/predicates.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface {

// Post-process smoothing by edge flips: repeatedly flip an interior edge when doing so lowers the
// total bend (the summed angle between adjacent faces' normals, 0 = flat) over its neighborhood.
// A flip changes the dihedral only on the five edges it touches (the diagonal and the quad's four
// sides), so that local drop equals the drop in the sum over every edge of the mesh -- the flips
// descend the mesh's total dihedral to a local minimum. Vertices are never moved, so every snapped
// point stays a vertex of the mesh; only the connectivity changes, which flattens the cusp/sliver
// artifacts a bad local triangulation produces.
//
// A priority queue drives the descent: each interior edge is keyed by the bend its flip would
// remove, and the best is taken first. A flip is applied against the current mesh (one at a time,
// so no two interfere), its stale neighbours re-scored and re-queued, and a popped edge is re-scored
// before use so an entry outdated by a nearby flip is simply skipped. The summed bend, bounded below
// by zero, strictly drops each flip, so the queue drains at a local minimum.
//
// All geometry is measured in the lattice's isotropic frame (aniso maps world into it) so the flip
// choice respects an anisotropic resolution; only the output keeps world positions. threshold
// (radians) gates which neighborhoods are touched: a flip is made only where the worst bend in its
// quad already exceeds threshold, so a region smoother than that is left exactly as generated.
// threshold = 0 flips wherever the summed bend can be lowered at all.
class Smoother {
  using Point2 = geometry::Point2;
  using Point3 = geometry::Point3;
  using Vector3 = geometry::Vector3;
  static constexpr double kPi = 3.141592653589793;

  // A candidate flip of edge {x, y} (faces f0i, f1i) into diagonal {c, d}, with the new triangles
  // and the total bend it removes (improve > 0).
  struct Flip {
    Index f0i;
    Index f1i;
    Index x;
    Index y;
    Index c;
    Index d;
    Face nf0;
    Face nf1;
    double improve;
  };

  // A queue entry: an edge keyed by the bend its flip removed when last scored (a hint that may be
  // stale; re-scored on pop).
  struct Item {
    double improve;
    Edge e;
  };
  struct ItemLess {
    bool operator()(const Item& a, const Item& b) const { return a.improve < b.improve; }
  };

  // A cell of the uniform spatial grid the self-intersection guard broad-phases over.
  using Cell = std::array<int, 3>;
  struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept {
      auto h = static_cast<std::size_t>(static_cast<std::uint32_t>(c[0])) * 73856093U;
      h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(c[1])) * 19349663U;
      h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(c[2])) * 83492791U;
      return h;
    }
  };

 public:
  // snapped, if non-empty, has one entry per vertex; a flip is then made only where its quad
  // touches a snapped vertex, leaving the rest of the mesh exactly as generated.
  Smoother(const Mesh& mesh, double resolution, const Mat3& aniso, double threshold,
           std::vector<bool> snapped = {})
      : v_(geometry::transform_points<3>(aniso, mesh.vertices())),
        f_(mesh.faces()),
        cell_(resolution),
        threshold_(threshold),
        snapped_(std::move(snapped)),
        visited_(f_.rows(), -1) {
    // The grid cell is the lattice resolution, the scale of an edge in this isotropic frame, so a
    // face spans about one cell and its AABB touches few. Cell size only tunes the broad-phase, not
    // its result, so this need not be exact -- but resolution is immune to a lone long edge that a
    // max-over-edges would let blow the grid up.
    for (Index fi = 0; fi < f_.rows(); fi++) {
      add_face_edges(fi);
      insert_face(fi);
    }

    std::priority_queue<Item, std::vector<Item>, ItemLess> pq;
    auto enqueue = [&](const Edge& e) {
      if (auto fl = score(e)) {
        pq.push({fl->improve, e});
      }
    };
    for (const auto& [e, faces] : ef_) {
      if (faces.size() == 2) {
        enqueue(e);
      }
    }

    std::int64_t flips = 0;
    auto cap = 50 * std::max<std::int64_t>(f_.rows(), 1);  // backstop against a float-noise cycle
    while (!pq.empty()) {
      Edge e = pq.top().e;
      pq.pop();
      auto fl = score(e);  // re-score: a nearby flip may have outdated this entry
      if (!fl || !guard_ok(*fl)) {
        continue;
      }
      do_flip(*fl);
      if (++flips > cap) {
        break;
      }
      // The flip changed only f0i and f1i, so re-score the edges those faces touch and the edges of
      // their neighbours (whose quads include a changed face).
      for (Index fi : {fl->f0i, fl->f1i}) {
        Face f = f_.row(fi);
        for (auto k = 0; k < 3; k++) {
          Edge ek{f[k], f[(k + 1) % 3]};
          enqueue(ek);
          if (auto it = ef_.find(ek); it != ef_.end()) {
            for (Index g : it->second) {
              if (g != fl->f0i && g != fl->f1i) {
                Face fg = f_.row(g);
                for (auto m = 0; m < 3; m++) {
                  enqueue(Edge{fg[m], fg[(m + 1) % 3]});
                }
              }
            }
          }
        }
      }
    }
    mesh_ = {mesh.vertices(), std::move(f_)};
  }

  Mesh mesh() && { return std::move(mesh_); }

 private:
  // The bend (0 = flat) between two faces; degenerate or back-to-back pairs read as the worst.
  double bend(const Face& a, const Face& b) const {
    Vector3 na = normal(a);
    Vector3 nb = normal(b);
    auto da = na.norm();
    auto db = nb.norm();
    if (!(da > 0.0) || !(db > 0.0)) {
      return kPi;
    }
    return std::acos(std::clamp(na.dot(nb) / (da * db), -1.0, 1.0));
  }

  // bend(a, face fi), or 0 when fi is the absent neighbour across a boundary edge (fi < 0).
  double bend_with(const Face& a, Index fi) const { return fi < 0 ? 0.0 : bend(a, f_.row(fi)); }

  // The interior face sharing e other than f0i/f1i, or -1 if e is a boundary edge.
  Index external(const Edge& e, Index f0i, Index f1i) const {
    auto it = ef_.find(e);
    if (it == ef_.end() || it->second.size() != 2) {
      return -1;
    }
    Index a = it->second[0];
    Index b = it->second[1];
    return a == f0i || a == f1i ? b : a;
  }

  // The candidate flip of edge e, if it is an interior edge whose flip is admissible (a new diagonal
  // that does not already exist, touches a snapped vertex, does not fold the surface back, and
  // strictly lowers the summed bend over a quad already creased past threshold). The cheap part of
  // the decision; the self-intersection guard is left to the caller, run only on the chosen flip.
  std::optional<Flip> score(const Edge& e) const {
    auto it = ef_.find(e);
    if (it == ef_.end() || it->second.size() != 2) {
      return std::nullopt;
    }
    Index f0i = it->second[0];
    Index f1i = it->second[1];
    Face f0 = f_.row(f0i);
    Face f1 = f_.row(f1i);

    Index x = -1;
    Index y = -1;
    for (auto k = 0; k < 3; k++) {
      if (e == Edge{f0[k], f0[(k + 1) % 3]}) {
        x = f0[k];
        y = f0[(k + 1) % 3];
        break;
      }
    }
    Index c = third(f0, e);
    Index d = third(f1, e);
    if (c < 0 || d < 0 || c == d || ef_.contains({c, d})) {
      return std::nullopt;  // boundary/degenerate, or the flipped diagonal already exists
    }
    if (!snapped_.empty() && !(snapped_[x] || snapped_[y] || snapped_[c] || snapped_[d])) {
      return std::nullopt;  // restricted to quads touching a snapped vertex
    }

    Face nf0{x, d, c};
    Face nf1{d, y, c};
    Vector3 nn0 = normal(nf0);
    Vector3 nn1 = normal(nf1);
    if (!(nn0.norm() > 0.0) || !(nn1.norm() > 0.0)) {
      return std::nullopt;
    }

    // Reject a flip that folds the surface back on itself: a new normal must not oppose the quad
    // normal (the sum of the two old face normals).
    Vector3 n0 = normal(f0);
    Vector3 n1 = normal(f1);
    auto d0 = n0.norm();
    auto d1 = n1.norm();
    if (d0 > 0.0 && d1 > 0.0) {
      Vector3 avg = n0 / d0 + n1 / d1;
      if (nn0.dot(avg) <= 0.0 || nn1.dot(avg) <= 0.0) {
        return std::nullopt;
      }
    }

    Index g_cx = external({c, x}, f0i, f1i);
    Index g_yc = external({y, c}, f0i, f1i);
    Index g_xd = external({x, d}, f0i, f1i);
    Index g_dy = external({d, y}, f0i, f1i);

    // Gate on the worst bend (smooth neighborhoods stay as generated) but decide on the summed bend
    // over the five touched edges, which descends the mesh's total dihedral.
    auto worst = std::max({bend(f0, f1), bend_with(f0, g_cx), bend_with(f0, g_yc),
                           bend_with(f1, g_xd), bend_with(f1, g_dy)});
    auto before = bend(f0, f1) + bend_with(f0, g_cx) + bend_with(f0, g_yc) + bend_with(f1, g_xd) +
                  bend_with(f1, g_dy);
    auto after = bend(nf0, nf1) + bend_with(nf0, g_cx) + bend_with(nf1, g_yc) + bend_with(nf0, g_xd) +
                 bend_with(nf1, g_dy);
    if (!(worst > threshold_) || !(after < before - 1e-6)) {
      return std::nullopt;
    }
    return Flip{f0i, f1i, x, y, c, d, nf0, nf1, before - after};
  }

  // Whether the flip keeps the mesh free of self-intersection: neither new triangle may cross any
  // face whose grid cell its AABB touches -- a spatial broad-phase, so a flipped diagonal that
  // passes over a topologically distant but spatially near sheet is caught, not just one in the
  // one-ring.
  bool guard_ok(const Flip& fl) {
    auto tol = 1e-6 * (v_.row(fl.x) - v_.row(fl.y)).norm();
    return !crosses(fl.nf0, fl.f0i, fl.f1i, tol) && !crosses(fl.nf1, fl.f0i, fl.f1i, tol);
  }

  // Whether new triangle nf (replacing f0i/f1i) overlaps any spatially near face. A face nf overlaps
  // has its AABB meet nf's, so scanning nf's own cells suffices; a fresh stamp per call tests each
  // such face against nf (the sibling call covers the other new triangle independently).
  bool crosses(const Face& nf, Index f0i, Index f1i, double tol) {
    guard_id_++;
    Point3 p0 = v_.row(nf[0]);
    Point3 p1 = v_.row(nf[1]);
    Point3 p2 = v_.row(nf[2]);
    Cell lo = cell_of(p0.cwiseMin(p1).cwiseMin(p2));
    Cell hi = cell_of(p0.cwiseMax(p1).cwiseMax(p2));
    for (auto i = lo[0]; i <= hi[0]; i++) {
      for (auto j = lo[1]; j <= hi[1]; j++) {
        for (auto k = lo[2]; k <= hi[2]; k++) {
          auto it = grid_.find(Cell{i, j, k});
          if (it == grid_.end()) {
            continue;
          }
          for (Index gi : it->second) {
            if (visited_[gi] == guard_id_) {
              continue;
            }
            visited_[gi] = guard_id_;
            if (gi != f0i && gi != f1i && overlaps(nf, f_.row(gi), tol)) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  // Apply the flip: rewrite the two faces in place and update the edge adjacency and spatial grid.
  void do_flip(const Flip& fl) {
    remove_face_edges(fl.f0i);
    remove_face_edges(fl.f1i);
    f_.row(fl.f0i) = fl.nf0;
    f_.row(fl.f1i) = fl.nf1;
    add_face_edges(fl.f0i);
    add_face_edges(fl.f1i);
    insert_face(fl.f0i);
    insert_face(fl.f1i);
  }

  Cell cell_of(const Point3& p) const {
    return {static_cast<int>(std::floor(p.x() / cell_)), static_cast<int>(std::floor(p.y() / cell_)),
            static_cast<int>(std::floor(p.z() / cell_))};
  }

  // Index a face by the grid cells its AABB touches. Vertices never move, so a face's cells change
  // only when it is flipped, when it is re-inserted in place; old entries left stale are harmless (a
  // stale index still reads its current geometry).
  void insert_face(Index fi) {
    Point3 p0 = v_.row(f_(fi, 0));
    Point3 p1 = v_.row(f_(fi, 1));
    Point3 p2 = v_.row(f_(fi, 2));
    Cell lo = cell_of(p0.cwiseMin(p1).cwiseMin(p2));
    Cell hi = cell_of(p0.cwiseMax(p1).cwiseMax(p2));
    for (auto i = lo[0]; i <= hi[0]; i++) {
      for (auto j = lo[1]; j <= hi[1]; j++) {
        for (auto k = lo[2]; k <= hi[2]; k++) {
          grid_[Cell{i, j, k}].push_back(fi);
        }
      }
    }
  }

  void add_face_edges(Index fi) {
    Face f = f_.row(fi);
    for (auto k = 0; k < 3; k++) {
      ef_[Edge{f[k], f[(k + 1) % 3]}].push_back(fi);
    }
  }

  void remove_face_edges(Index fi) {
    Face f = f_.row(fi);
    for (auto k = 0; k < 3; k++) {
      auto it = ef_.find(Edge{f[k], f[(k + 1) % 3]});
      if (it == ef_.end()) {
        continue;
      }
      auto& faces = it->second;
      faces.erase(std::remove(faces.begin(), faces.end(), fi), faces.end());
      if (faces.empty()) {
        ef_.erase(it);
      }
    }
  }

  // A face's unnormalized normal (length is twice the area).
  Vector3 normal(const Face& t) const {
    Vector3 e1 = v_.row(t[1]) - v_.row(t[0]);
    Vector3 e2 = v_.row(t[2]) - v_.row(t[0]);
    return Vector3(e1.cross(e2));
  }

  // A real crossing of a and b beyond a shared vertex. Edge-adjacent pairs (shared >= 2) are
  // skipped: a flip cannot fold the surface back on an edge without either opposing the quad normal
  // (rejected above) or worsening a quad bend (so it is not an improvement and is not made), so the
  // only crossings left to catch are with non-adjacent faces.
  bool overlaps(const Face& a, const Face& b, double tol) const {
    auto shared = shared_vertices(a, b);
    if (shared >= 2) {
      return false;
    }
    Point3 a0 = v_.row(a[0]);
    Point3 a1 = v_.row(a[1]);
    Point3 a2 = v_.row(a[2]);
    Point3 b0 = v_.row(b[0]);
    Point3 b1 = v_.row(b[1]);
    Point3 b2 = v_.row(b[2]);
    if (shared == 0) {
      // A disjoint pair gets the robust exact crossing test (no false negatives, unlike the
      // shrink-based transversal in triangles_overlap_3d -- a flip can pass a new face over a
      // non-adjacent sheet that the shrink test would miss).
      return triangles_cross_3d(a0, a1, a2, b0, b1, b2);
    }
    return triangles_overlap_3d(a0, a1, a2, b0, b1, b2, tol);
  }

  static int shared_vertices(const Face& a, const Face& b) {
    auto n = 0;
    for (auto u : a) {
      for (auto w : b) {
        if (u == w) {
          n++;
        }
      }
    }
    return n;
  }

  // The vertex of f that is neither endpoint of e, or -1 if there is none.
  static Index third(const Face& f, const Edge& e) {
    for (auto x : f) {
      if (x != e.a && x != e.b) {
        return x;
      }
    }
    return -1;
  }

  geometry::Points3 v_;  // vertices in the isotropic frame, where the geometry is measured
  Faces f_;              // working faces; connectivity is edited in place
  double cell_{};        // spatial-grid cell size for the self-intersection guard
  double threshold_;
  std::vector<bool> snapped_;  // if non-empty, restrict flips to quads touching a snapped vertex
  std::unordered_map<Edge, std::vector<Index>, EdgeHash> ef_;     // edge -> its (<= 2) incident faces
  std::unordered_map<Cell, std::vector<Index>, CellHash> grid_;   // spatial broad-phase for the guard
  std::vector<Index> visited_;  // per-guard stamp, to test each candidate face once
  Index guard_id_{0};
  Mesh mesh_;
};

// The post-snap cleanup: flatten the snapped region by edge flips (see Smoother), so that the
// cusp/sliver artifacts the snapped triangulation leaves are smoothed wherever a crease exceeds
// 5 degrees, while the rest of the mesh is left exactly as generated. aniso maps world into the
// isotropic frame and resolution is the lattice spacing in it (the broad-phase grid cell). snapped,
// if non-empty (one entry per vertex), restricts flips to quads touching a snapped vertex; when
// empty, nothing is flipped. Vertices never move, so every snap point stays.
inline Mesh post_snap(const Mesh& mesh, double resolution, const Mat3& aniso,
                      std::vector<bool> snapped = {}) {
  constexpr double kDegree = 0.017453292519943295;  // radians
  return Smoother(mesh, resolution, aniso, 5.0 * kDegree, std::move(snapped)).mesh();
}

}  // namespace polatory::isosurface
