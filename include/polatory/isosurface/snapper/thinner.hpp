#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/predicates.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace polatory::isosurface::snapper {

using geometry::Points3;

// Cross-pass thinning by guarded edge collapse. A snap point captured in one pass can become
// redundant only after a neighbour is captured in a later pass (e.g. it ends up collinear between
// two others); by then it is an ordinary vertex of the later pass's input, beyond the reach of that
// pass's insert-only thinning (see thin_inserted in Snapper). This final sweep collapses such a
// redundant snapped vertex onto a neighbour -- the same vertex removal, expressed on the plain mesh
// so it reaches vertices from any pass. A collapse is kept only when the dropped point stays within
// its snapping tolerance of the new surface and the result stays manifold, unflipped, and free of
// self-intersection. Only snapped vertices are collapsed, so the base lattice is left as generated.
// All geometry is measured in the lattice's isotropic frame (aniso maps world into it); the output
// keeps world positions.
class Thinner {
  using Point3 = geometry::Point3;
  using Vector3 = geometry::Vector3;
  using Face = std::array<Index, 3>;

 public:
  Thinner(const Points3& vertices, const Faces& faces, const Points3& points,
          const VecX& tolerances, const Mat3& aniso) {
    auto iso = geometry::transform_points<3>(aniso, vertices);
    world_.assign(vertices.rowwise().begin(), vertices.rowwise().end());
    iso_.assign(iso.rowwise().begin(), iso.rowwise().end());
    faces_.reserve(faces.rows());
    for (Index i = 0; i < faces.rows(); i++) {
      faces_.push_back({faces(i, 0), faces(i, 1), faces(i, 2)});
    }
    deleted_.assign(faces_.size(), false);

    // Each vertex's snapping tolerance, by exact match to a snap point (a snapped vertex is emitted
    // at the point's exact world position); -1 marks a vertex that is not a snap point, never moved.
    std::unordered_map<Point3, double, PointHash> point_tol;
    point_tol.reserve(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      point_tol[points.row(i)] = tolerances.size() != 0 ? tolerances(i) : 0.0;
    }
    tol_.assign(world_.size(), -1.0);
    for (std::size_t v = 0; v < world_.size(); v++) {
      if (auto it = point_tol.find(world_.at(v)); it != point_tol.end()) {
        tol_.at(v) = it->second;
      }
    }
  }

  Mesh thin() {
    v2f_.assign(world_.size(), {});
    for (std::size_t fi = 0; fi < faces_.size(); fi++) {
      for (auto v : faces_.at(fi)) {
        v2f_.at(v).push_back(static_cast<Index>(fi));
      }
    }
    // Greedy to a fixpoint: a collapse can make a neighbour collapsible (a chain of collinear
    // points thins end to end).
    bool any = true;
    while (any) {
      any = false;
      for (Index v = 0; v < static_cast<Index>(world_.size()); v++) {
        if (tol_.at(v) >= 0.0 && try_collapse(v)) {
          any = true;
        }
      }
    }
    return emit();
  }

 private:
  struct PointHash {
    std::size_t operator()(const Point3& p) const noexcept {
      std::hash<double> h;
      return h(p.x()) ^ (h(p.y()) << 1) ^ (h(p.z()) << 2);
    }
  };

  // The (alive) faces incident to v.
  std::vector<Index> incident(Index v) const {
    std::vector<Index> fs;
    for (auto fi : v2f_.at(v)) {
      if (!deleted_.at(fi)) {
        fs.push_back(fi);
      }
    }
    return fs;
  }

  Vector3 normal(const Face& f) const {
    return Vector3((iso_.at(f[1]) - iso_.at(f[0])).cross(iso_.at(f[2]) - iso_.at(f[0])));
  }

  // Whether faces a and b intersect with positive measure (a real self-intersection).
  bool intersects(const Face& a, const Face& b) const {
    return triangles_intersect(iso_.at(a[0]), iso_.at(a[1]), iso_.at(a[2]), iso_.at(b[0]),
                               iso_.at(b[1]), iso_.at(b[2]), shared_vertices(a, b));
  }

  // Collapse the snapped vertex v onto its least-distorting admissible neighbour, if any. Returns
  // whether a collapse was made.
  bool try_collapse(Index v) {
    auto inc = incident(v);
    if (inc.size() < 3) {
      return false;
    }
    // The neighbours of v, and the faces on each edge (v, w).
    std::unordered_map<Index, std::vector<Index>> edge_faces;
    for (auto fi : inc) {
      for (auto w : faces_.at(fi)) {
        if (w != v) {
          edge_faces[w].push_back(fi);
        }
      }
    }
    // v must be an interior manifold vertex: every spoke shared by exactly two incident faces.
    for (const auto& [w, fs] : edge_faces) {
      if (fs.size() != 2) {
        return false;
      }
    }

    Index best = -1;
    double best_dev = std::numeric_limits<double>::infinity();
    for (const auto& [w, fs] : edge_faces) {
      double dev = 0.0;
      if (collapse_ok(v, w, inc, edge_faces, dev) && dev < best_dev) {
        best = w;
        best_dev = dev;
      }
    }
    if (best < 0) {
      return false;
    }
    do_collapse(v, best, inc, edge_faces.at(best));
    return true;
  }

  // Whether collapsing v onto w keeps the mesh manifold, unflipped, free of self-intersection, and
  // the dropped point v within its tolerance of the new surface. dev returns that distance.
  bool collapse_ok(Index v, Index w, const std::vector<Index>& inc,
                   const std::unordered_map<Index, std::vector<Index>>& edge_faces, double& dev) {
    // Link condition: the only vertices adjacent to both v and w must be the two opposite the edge
    // (v, w); otherwise the collapse folds two sheets together into a non-manifold edge.
    std::unordered_set<Index> across;
    for (auto fi : edge_faces.at(w)) {
      for (auto x : faces_.at(fi)) {
        if (x != v && x != w) {
          across.insert(x);
        }
      }
    }
    for (const auto& [x, fs] : edge_faces) {
      if (x == w) {
        continue;
      }
      // x is a neighbour of v; is it also a neighbour of w through a face not on edge (v, w)?
      bool adj_w = false;
      for (auto fi : v2f_.at(x)) {
        if (deleted_.at(fi)) {
          continue;
        }
        const auto& f = faces_.at(fi);
        if ((f[0] == w || f[1] == w || f[2] == w) && !on_edge(f, v, w)) {
          adj_w = true;
          break;
        }
      }
      if (adj_w && !across.contains(x)) {
        return false;
      }
    }

    // The faces v keeps (not on edge (v, w)), with v retargeted to w.
    std::vector<Face> kept;
    std::unordered_set<Index> umbrella(inc.begin(), inc.end());
    for (auto fi : inc) {
      const auto& f = faces_.at(fi);
      if (on_edge(f, v, w)) {
        continue;  // collapses to a degenerate sliver, dropped
      }
      Face nf{f[0] == v ? w : f[0], f[1] == v ? w : f[1], f[2] == v ? w : f[2]};
      Vector3 nn = normal(nf);
      if (!(nn.norm() > 0.0)) {
        return false;  // a kept face would become degenerate
      }
      if (nn.dot(normal(f)) <= 0.0) {
        return false;  // the face would flip
      }
      kept.push_back(nf);
    }
    if (kept.empty()) {
      return false;
    }

    // The dropped point must stay within its tolerance of the new surface.
    double best = std::numeric_limits<double>::infinity();
    for (const auto& nf : kept) {
      best = std::min(best, point_triangle_dist2(iso_.at(v), iso_.at(nf[0]), iso_.at(nf[1]),
                                                 iso_.at(nf[2])));
    }
    if (best > tol_.at(v) * tol_.at(v)) {
      return false;
    }
    dev = best;

    // No new face may self-intersect a nearby face (one incident to a ring vertex, outside the
    // umbrella). A collapse is local, so the one-ring neighborhood suffices.
    std::unordered_set<Index> nearby;
    for (const auto& nf : kept) {
      for (auto x : nf) {
        for (auto fi : v2f_.at(x)) {
          if (!deleted_.at(fi) && !umbrella.contains(fi)) {
            nearby.insert(fi);
          }
        }
      }
    }
    for (const auto& nf : kept) {
      for (auto fi : nearby) {
        if (intersects(nf, faces_.at(fi))) {
          return false;
        }
      }
    }
    return true;
  }

  void do_collapse(Index v, Index w, const std::vector<Index>& inc,
                   const std::vector<Index>& vw_faces) {
    std::unordered_set<Index> dropped(vw_faces.begin(), vw_faces.end());
    for (auto fi : inc) {
      if (dropped.contains(fi)) {
        deleted_.at(fi) = true;
        continue;
      }
      auto& f = faces_.at(fi);
      for (auto& x : f) {
        if (x == v) {
          x = w;
        }
      }
      v2f_.at(w).push_back(fi);  // w gains v's retargeted faces
    }
  }

  static bool on_edge(const Face& f, Index a, Index b) {
    bool ha = f[0] == a || f[1] == a || f[2] == a;
    bool hb = f[0] == b || f[1] == b || f[2] == b;
    return ha && hb;
  }

  Mesh emit() {
    std::vector<bool> used(world_.size(), false);
    std::vector<Face> faces;
    for (std::size_t fi = 0; fi < faces_.size(); fi++) {
      if (deleted_.at(fi)) {
        continue;
      }
      faces.push_back(faces_.at(fi));
      for (auto v : faces_.at(fi)) {
        used.at(v) = true;
      }
    }
    std::vector<Index> remap(world_.size(), -1);
    Index n = 0;
    for (std::size_t v = 0; v < world_.size(); v++) {
      if (used.at(v)) {
        remap.at(v) = n++;
      }
    }
    Points3 vertices(n, 3);
    for (std::size_t v = 0; v < world_.size(); v++) {
      if (used.at(v)) {
        vertices.row(remap.at(v)) = world_.at(v);
      }
    }
    Faces f(static_cast<Index>(faces.size()), 3);
    for (Index i = 0; i < f.rows(); i++) {
      f(i, 0) = remap.at(faces.at(i)[0]);
      f(i, 1) = remap.at(faces.at(i)[1]);
      f(i, 2) = remap.at(faces.at(i)[2]);
    }
    return {std::move(vertices), std::move(f)};
  }

  std::vector<Point3> world_;
  std::vector<Point3> iso_;
  std::vector<Face> faces_;
  std::vector<bool> deleted_;
  std::vector<double> tol_;                  // per vertex; -1 = not a snap point (never collapsed)
  std::vector<std::vector<Index>> v2f_;      // vertex -> incident face ids (may include deleted)
};

}  // namespace polatory::isosurface::snapper
