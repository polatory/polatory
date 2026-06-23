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
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "spatial_grid.hpp"
#include "utility.hpp"

namespace polatory::isosurface::snapper {

using geometry::Points3;

// Cross-pass thinning by guarded edge collapse: collapses a snapped vertex that a later pass left
// redundant (collinear between neighbours) onto a neighbour, reaching vertices from any pass that
// the insert-only thinning could not. A collapse is kept only if the dropped point stays within its
// tolerance of the new surface and the mesh stays manifold, unflipped, and self-intersection-free;
// only snapped vertices collapse, so the base lattice is untouched. Geometry is in the isotropic
// frame; the output keeps world positions.
class Thinner {
  using Point3 = geometry::Point3;
  using Vector3 = geometry::Vector3;
  static constexpr double kMaxEdgeRatio =
      1.5;  // a collapse may not make an edge longer than this * res

 public:
  Thinner(const Mesh& mesh, const Points3& points, const VecX& tolerances, double resolution,
          const Mat3& aniso)
      : snap_points_(geometry::transform_points<3>(aniso, points)),
        snap_tols_(tolerances),
        snap_grid_(resolution, points.rows()),
        max_edge2_(kMaxEdgeRatio * resolution * (kMaxEdgeRatio * resolution)) {
    auto iso = geometry::transform_points<3>(aniso, mesh.vertices());
    world_.assign(mesh.vertices().rowwise().begin(), mesh.vertices().rowwise().end());
    iso_.assign(iso.rowwise().begin(), iso.rowwise().end());
    const auto& faces = mesh.faces();
    faces_.reserve(faces.rows());
    for (Index i = 0; i < faces.rows(); i++) {
      faces_.push_back({faces(i, 0), faces(i, 1), faces(i, 2)});
    }
    deleted_.assign(faces_.size(), false);

    if (snap_tols_.size() == 0) {
      snap_tols_ = VecX::Zero(snap_points_.rows());
    }
    // Insert each snap point as a tolerance-radius ball, so a query AABB finds every point it
    // reaches.
    for (Index i = 0; i < snap_points_.rows(); i++) {
      Vector3 r = Vector3::Constant(snap_tols_(i));
      snap_grid_.insert(i, snap_points_.row(i) - r, snap_points_.row(i) + r);
    }

    // Per-vertex tolerance by exact match to a snap point (snapped vertices are emitted there);
    // -1 = not a snap point, never collapsed.
    std::unordered_map<Point3, double, PointHash> point_tol;
    point_tol.reserve(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      point_tol[points.row(i)] = snap_tols_(i);
    }
    tol_.assign(world_.size(), -1.0);
    for (std::size_t v = 0; v < world_.size(); v++) {
      if (auto it = point_tol.find(world_.at(v)); it != point_tol.end()) {
        tol_.at(v) = it->second;
      }
    }

    // Greedy collapse to a fixpoint: a collapse can make a neighbour collapsible (a chain of
    // collinear points thins end to end).
    v2f_.assign(world_.size(), {});
    for (std::size_t fi = 0; fi < faces_.size(); fi++) {
      for (auto v : faces_.at(fi)) {
        v2f_.at(v).push_back(static_cast<Index>(fi));
      }
    }
    bool any = true;
    while (any) {
      any = false;
      for (Index v = 0; v < static_cast<Index>(world_.size()); v++) {
        if (tol_.at(v) >= 0.0 && try_collapse(v)) {
          any = true;
        }
      }
    }
    result_ = emit();
  }

  Mesh result() && { return std::move(result_); }

 private:
  struct PointHash {
    std::size_t operator()(const Point3& p) const noexcept {
      std::hash<double> h;
      return h(p.x()) ^ (h(p.y()) << 1) ^ (h(p.z()) << 2);
    }
  };

  // Whether collapsing v onto w is admissible; dev = the dropped point's distance to the new
  // surface.
  bool collapse_ok(Index v, Index w, const std::vector<Index>& inc,
                   const std::unordered_map<Index, std::vector<Index>>& edge_faces, double& dev) {
    // Link condition: v and w may share only the two vertices opposite edge (v, w), else the
    // collapse folds two sheets into a non-manifold edge.
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
      bool adj_w = false;
      for (auto fi : v2f_.at(x)) {
        if (deleted_.at(fi)) {
          continue;
        }
        const auto& f = faces_.at(fi);
        if ((f(0) == w || f(1) == w || f(2) == w) && !on_edge(f, v, w)) {
          adj_w = true;
          break;
        }
      }
      if (adj_w && !across.contains(x)) {
        return false;
      }
    }

    // v's kept faces (those not on edge (v, w)), with v retargeted to w.
    std::vector<Face> kept;
    std::unordered_set<Index> umbrella(inc.begin(), inc.end());
    for (auto fi : inc) {
      const auto& f = faces_.at(fi);
      if (on_edge(f, v, w)) {
        continue;  // collapses to a degenerate sliver, dropped
      }
      Face nf{f(0) == v ? w : f(0), f(1) == v ? w : f(1), f(2) == v ? w : f(2)};
      auto nn = normal(nf);
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

    // Cap edge length to keep triangles regular: a collapse stretches only v's spokes (v-r -> w-r),
    // so only those are checked.
    for (const auto& [r, fs] : edge_faces) {
      if (r != w && !across.contains(r) && (iso_.at(w) - iso_.at(r)).squaredNorm() > max_edge2_) {
        return false;
      }
    }

    // Distortion (for picking the least-distorting neighbour): the dropped vertex's distance to the
    // new surface.
    dev = std::numeric_limits<double>::infinity();
    for (const auto& nf : kept) {
      dev = std::min(dev, dist2(iso_.at(v), nf));
    }

    // Faces near the collapse: those incident to the kept faces' vertices but outside the umbrella.
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

    if (!honors_ok(inc, kept, nearby)) {
      return false;
    }

    // No new face may self-intersect a nearby face. A collapse is local, so the one-ring suffices.
    for (const auto& nf : kept) {
      for (auto fi : nearby) {
        if (intersects(nf, faces_.at(fi))) {
          return false;
        }
      }
    }
    return true;
  }

  double dist2(const Point3& p, const Face& f) const {
    return point_triangle_dist2(p, iso_.at(f(0)), iso_.at(f(1)), iso_.at(f(2)));
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
      f(i, 0) = remap.at(faces.at(i)(0));
      f(i, 1) = remap.at(faces.at(i)(1));
      f(i, 2) = remap.at(faces.at(i)(2));
    }
    return {std::move(vertices), std::move(f)};
  }

  // Every nearby snap point held by a removed face -- not just the dropped vertex -- must stay
  // within tolerance, else a greedy chain drifts already-dropped points off the surface.
  bool honors_ok(const std::vector<Index>& inc, const std::vector<Face>& kept,
                 const std::unordered_set<Index>& nearby) const {
    if (snap_grid_.empty()) {
      return true;
    }
    Point3 lo = iso_.at(faces_.at(inc.front())(0));
    Point3 hi = lo;
    for (auto fi : inc) {
      for (auto x : faces_.at(fi)) {
        lo = lo.cwiseMin(iso_.at(x));
        hi = hi.cwiseMax(iso_.at(x));
      }
    }
    bool ok = true;
    snap_grid_.for_each(lo, hi, [&](Index pi) {
      auto t2 = snap_tols_(pi) * snap_tols_(pi);
      Point3 p = snap_points_.row(pi);
      auto old = std::numeric_limits<double>::infinity();
      for (auto fi : inc) {
        old = std::min(old, dist2(p, faces_.at(fi)));
      }
      if (old > t2) {
        return true;  // not held by a removed face; the collapse cannot dishonor it
      }
      auto neu = std::numeric_limits<double>::infinity();
      for (const auto& nf : kept) {
        neu = std::min(neu, dist2(p, nf));
      }
      for (auto fi : nearby) {
        neu = std::min(neu, dist2(p, faces_.at(fi)));
      }
      if (neu > t2) {
        ok = false;
        return false;  // dishonored; stop the walk
      }
      return true;
    });
    return ok;
  }

  std::vector<Index> incident(Index v) const {
    std::vector<Index> fs;
    for (auto fi : v2f_.at(v)) {
      if (!deleted_.at(fi)) {
        fs.push_back(fi);
      }
    }
    return fs;
  }

  bool intersects(const Face& a, const Face& b) const {
    return triangles_intersect(iso_.at(a(0)), iso_.at(a(1)), iso_.at(a(2)), iso_.at(b(0)),
                               iso_.at(b(1)), iso_.at(b(2)), num_shared_vertices(a, b));
  }

  Vector3 normal(const Face& f) const {
    return Vector3((iso_.at(f(1)) - iso_.at(f(0))).cross(iso_.at(f(2)) - iso_.at(f(0))));
  }

  static bool on_edge(const Face& f, Index a, Index b) {
    bool ha = f(0) == a || f(1) == a || f(2) == a;
    bool hb = f(0) == b || f(1) == b || f(2) == b;
    return ha && hb;
  }

  // Collapse v onto its least-distorting admissible neighbour, if any; returns whether it did.
  bool try_collapse(Index v) {
    auto inc = incident(v);
    if (inc.size() < 3) {
      return false;
    }
    std::unordered_map<Index, std::vector<Index>>
        edge_faces;  // neighbour w -> faces on edge (v, w)
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

  Points3 snap_points_;  // snap targets in the isotropic frame
  VecX snap_tols_;       // snapping tolerance per snap point (isotropic-frame distance)
  SpatialGrid snap_grid_;
  double max_edge2_{};  // squared cap on a collapsed edge's length
  std::vector<Point3> world_;
  std::vector<Point3> iso_;
  std::vector<Face> faces_;
  std::vector<bool> deleted_;
  std::vector<double> tol_;              // per vertex; -1 = not a snap point (never collapsed)
  std::vector<std::vector<Index>> v2f_;  // vertex -> incident face ids (may include deleted)
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
