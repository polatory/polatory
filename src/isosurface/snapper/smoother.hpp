#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <queue>
#include <utility>
#include <vector>

#include "abstract_mesh.hpp"
#include "spatial_grid.hpp"
#include "utility.hpp"

namespace polatory::isosurface::snapper {

// Post-process smoothing by edge flips: flip an interior edge while it lowers the total bend
// (summed dihedral). A flip changes only its five edges, so the local drop equals the global drop
// and the mesh descends to a local minimum; vertices never move, so snapped points stay vertices. A
// priority queue takes the largest improvement first, re-scoring each popped edge (a nearby flip
// may have staled it). Geometry is in the isotropic frame; the output keeps world positions. A flip
// is rejected if its new diagonal overshoots the bend-dependent length cap (see kEdgeFloor), folds
// the surface, or pushes it beyond a snap tolerance (protecting points honored within tolerance
// with no vertex there).
class Smoother {
  using Point2 = geometry::Point2;
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using Vector3 = geometry::Vector3;
  static constexpr double kPi = 3.141592653589793;
  // Dihedral-dependent length cap. A flip's new diagonal may always reach kEdgeFloor * res; beyond
  // that each unit of overshoot (in res) must be paid for by kImproveFull / (kEdgeCeiling -
  // kEdgeFloor) radians of bend reduction, and kEdgeCeiling * res is the hard ceiling. So a long
  // diagonal is admitted only when it flattens a genuine crease, not a flat-direction (cosmetic)
  // one.
  static constexpr double kEdgeFloor = 1.5;
  static constexpr double kEdgeCeiling = 2.0;
  static constexpr double kImproveFull = kPi / 2;  // bend reduction earning the full ceiling

  // A candidate flip of edge {x, y} (faces fi0, fi1) into diagonal {c, d}; improve > 0 is the total
  // bend removed.
  struct Flip {
    Index fi0;
    Index fi1;
    Index x;
    Index y;
    Index c;
    Index d;
    double improve;

    Face new_f0() const { return {c, x, d}; }
    Face new_f1() const { return {d, y, c}; }
  };

  // A queue entry: an edge keyed by the bend its flip removed when last scored (a hint that may be
  // stale; re-scored on pop).
  struct Item {
    Edge e;
    double improve;
  };

  struct ItemLess {
    bool operator()(const Item& a, const Item& b) const { return a.improve < b.improve; }
  };

 public:
  Smoother(const Mesh& mesh, const Points3& points, const VecX& tolerances, double resolution,
           const Mat3& aniso, double min_angle)
      : v_(geometry::transform_points<3>(aniso, mesh.vertices())),
        mesh_(mesh.faces()),
        snap_points_(geometry::transform_points<3>(aniso, points)),
        snap_tols_(tolerances),
        snap_grid_(resolution, points.rows()),
        face_grid_(resolution, mesh_.num_faces()),
        resolution_(resolution),
        min_angle_(min_angle) {
    // Grid cell = resolution (a face spans about one cell); it only tunes the broad-phase, and
    // resolution avoids a lone long edge blowing the grid up.
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      insert_face(fi);
    }

    if (snap_tols_.size() == 0) {
      snap_tols_ = VecX::Zero(snap_points_.rows());
    }
    // Insert each snap point as a tolerance-radius ball, so a query AABB finds every point it
    // reaches.
    for (Index i = 0; i < snap_points_.rows(); i++) {
      Vector3 r = Vector3::Constant(snap_tols_(i));
      snap_grid_.insert(i, snap_points_.row(i) - r, snap_points_.row(i) + r);
    }

    std::priority_queue<Item, std::vector<Item>, ItemLess> pq;
    auto enqueue = [&](const Edge& e) {
      if (auto fl = score(e)) {
        pq.push({e, fl->improve});
      }
    };
    mesh_.for_each_edge([&](const Edge& e, const auto& sides) {
      if (sides[0] >= 0 && sides[1] >= 0) {
        enqueue(e);
      }
    });

    std::int64_t flips = 0;
    auto cap = 50 * std::max<std::int64_t>(mesh_.num_faces(), 1);  // backstop against a float cycle
    while (!pq.empty()) {
      Edge e = pq.top().e;
      pq.pop();
      auto fl = score(e);  // re-score: a nearby flip may have outdated this entry
      if (!fl || !honors_ok(*fl) || !guard_ok(*fl)) {
        continue;
      }
      do_flip(*fl);
      if (++flips > cap) {
        break;
      }
      // Re-score the edges of the two changed faces and of their neighbours.
      for (Index fi : {fl->fi0, fl->fi1}) {
        auto f = mesh_.face(fi);
        for (auto k = 0; k < 3; k++) {
          Edge ek{f(k), f((k + 1) % 3)};
          enqueue(ek);
          Index gi = mesh_.across(ek, fi);
          if (gi >= 0 && gi != fl->fi0 && gi != fl->fi1) {
            auto g = mesh_.face(gi);
            for (auto m = 0; m < 3; m++) {
              enqueue({g(m), g((m + 1) % 3)});
            }
          }
        }
      }
    }
    result_ = {mesh.vertices(), std::move(mesh_).take_faces()};
  }

  Mesh result() && { return std::move(result_); }

 private:
  // The bend (0 = flat) between two faces; degenerate or back-to-back pairs read as the worst.
  double bend(const Face& a, const Face& b) const {
    auto na = normal(a);
    auto nb = normal(b);
    auto da = na.norm();
    auto db = nb.norm();
    if (!(da > 0.0) || !(db > 0.0)) {
      return kPi;
    }
    return std::acos(std::clamp(na.dot(nb) / (da * db), -1.0, 1.0));
  }

  // bend(a, face fi), or 0 when fi is the absent neighbour across a boundary edge (fi < 0).
  double bend_with(const Face& a, Index fi) const { return fi < 0 ? 0.0 : bend(a, mesh_.face(fi)); }

  // Whether new_f overlaps any spatially near face; scanning new_f's own cells suffices (the
  // sibling call covers the other new triangle).
  bool crosses(const Face& new_f, Index fi0, Index fi1, double tol) {
    Point3 p0 = v_.row(new_f(0));
    Point3 p1 = v_.row(new_f(1));
    Point3 p2 = v_.row(new_f(2));
    Point3 lo = p0.cwiseMin(p1).cwiseMin(p2);
    Point3 hi = p0.cwiseMax(p1).cwiseMax(p2);
    bool hit = false;
    face_grid_.for_each(lo, hi, [&](Index gi) {
      if (gi != fi0 && gi != fi1 && overlaps(new_f, mesh_.face(gi), tol)) {
        hit = true;
        return false;
      }
      return true;
    });
    return hit;
  }

  double dist2(const Point3& p, const Face& f) const {
    return point_triangle_dist2(p, v_.row(f(0)), v_.row(f(1)), v_.row(f(2)));
  }

  void do_flip(const Flip& fl) {
    auto [fi0, fi1] = mesh_.flip({fl.x, fl.y});
    insert_face(fi0);
    insert_face(fi1);
  }

  // Self-intersection guard: neither new triangle may cross a face in its grid cells -- a
  // broad-phase that catches a diagonal passing over a spatially near but topologically distant
  // sheet.
  bool guard_ok(const Flip& fl) {
    auto tol = 1e-6 * (v_.row(fl.x) - v_.row(fl.y)).norm();
    return !crosses(fl.new_f0(), fl.fi0, fl.fi1, tol) && !crosses(fl.new_f1(), fl.fi0, fl.fi1, tol);
  }

  // A point within tolerance of a removed face must stay within tolerance of the new local faces
  // (the two new triangles plus the quad's outer neighbours); only such points can be affected.
  bool honors_ok(const Flip& fl) const {
    if (snap_grid_.empty()) {
      return true;
    }
    auto f0 = mesh_.face(fl.fi0);
    auto f1 = mesh_.face(fl.fi1);
    std::array<Face, 6> after{fl.new_f0(), fl.new_f1()};
    auto na = 2;
    // Each outer edge belongs to one removed face; its neighbour across that face also bends.
    auto add_neighbour = [&](const Edge& e, Index owner) {
      Index gi = mesh_.across(e, owner);
      if (gi >= 0) {
        after.at(na++) = mesh_.face(gi);
      }
    };
    add_neighbour({fl.c, fl.x}, fl.fi0);
    add_neighbour({fl.y, fl.c}, fl.fi0);
    add_neighbour({fl.x, fl.d}, fl.fi1);
    add_neighbour({fl.d, fl.y}, fl.fi1);
    Point3 lo = v_.row(fl.x).cwiseMin(v_.row(fl.y)).cwiseMin(v_.row(fl.c)).cwiseMin(v_.row(fl.d));
    Point3 hi = v_.row(fl.x).cwiseMax(v_.row(fl.y)).cwiseMax(v_.row(fl.c)).cwiseMax(v_.row(fl.d));
    bool ok = true;
    snap_grid_.for_each(lo, hi, [&](Index pi) {
      auto tol2 = snap_tols_(pi) * snap_tols_(pi);
      Point3 p = snap_points_.row(pi);
      if (std::min(dist2(p, f0), dist2(p, f1)) > tol2) {
        return true;  // not honored by a removed face; the flip cannot dishonor it
      }
      auto best = std::numeric_limits<double>::infinity();
      for (auto m = 0; m < na; m++) {
        best = std::min(best, dist2(p, after.at(m)));
      }
      if (best > tol2) {
        ok = false;
        return false;  // dishonored; stop the walk
      }
      return true;
    });
    return ok;
  }

  // Index a face by the grid cells its AABB touches; stale entries from a flip are harmless (a
  // stale index reads its current geometry).
  void insert_face(Index fi) {
    auto f = mesh_.face(fi);
    Point3 p0 = v_.row(f(0));
    Point3 p1 = v_.row(f(1));
    Point3 p2 = v_.row(f(2));
    face_grid_.insert(fi, p0.cwiseMin(p1).cwiseMin(p2), p0.cwiseMax(p1).cwiseMax(p2));
  }

  double min_angle(const Face& f) const {
    return triangle_min_angle(v_.row(f(0)), v_.row(f(1)), v_.row(f(2)));
  }

  // A face's unnormalized normal (length is twice the area).
  Vector3 normal(const Face& f) const {
    Vector3 e1 = v_.row(f(1)) - v_.row(f(0));
    Vector3 e2 = v_.row(f(2)) - v_.row(f(0));
    return Vector3(e1.cross(e2));
  }

  // A real crossing beyond a shared vertex. Edge-adjacent pairs (shared >= 2) are skipped: such a
  // fold would oppose the quad normal or worsen the bend, so it is rejected elsewhere.
  bool overlaps(const Face& a, const Face& b, double tol) const {
    auto shared = num_shared_vertices(a, b);
    if (shared >= 2) {
      return false;
    }
    Point3 a0 = v_.row(a(0));
    Point3 a1 = v_.row(a(1));
    Point3 a2 = v_.row(a(2));
    Point3 b0 = v_.row(b(0));
    Point3 b1 = v_.row(b(1));
    Point3 b2 = v_.row(b(2));
    if (shared == 0) {
      // Disjoint pair: the exact crossing test (no false negatives, unlike the shrink-based
      // transversal) catches a new face passing over a non-adjacent sheet.
      return triangles_cross_3d(a0, a1, a2, b0, b1, b2);
    }
    return triangles_overlap_3d(a0, a1, a2, b0, b1, b2, tol);
  }

  // The candidate flip of edge e if admissible (interior, new diagonal absent, within the length
  // cap, no fold, lowers the bend). The cheap checks; the self-int guard is left to the caller.
  std::optional<Flip> score(const Edge& e) const {
    auto [fi0, fi1] = mesh_.faces_of(e);
    if (fi0 < 0 || fi1 < 0) {
      return std::nullopt;
    }
    auto f0 = mesh_.face(fi0);
    auto f1 = mesh_.face(fi1);

    Index x = -1;
    Index y = -1;
    for (auto k = 0; k < 3; k++) {
      if (e == Edge{f0(k), f0((k + 1) % 3)}) {
        x = f0(k);
        y = f0((k + 1) % 3);
        break;
      }
    }
    Index c = AbstractMesh::opposite(f0, e);
    Index d = AbstractMesh::opposite(f1, e);
    if (c < 0 || d < 0 || c == d || mesh_.has_edge({c, d})) {
      return std::nullopt;  // boundary/degenerate, or the flipped diagonal already exists
    }
    auto new_len2 = (v_.row(c) - v_.row(d)).squaredNorm();
    auto cur_len2 = (v_.row(x) - v_.row(y)).squaredNorm();
    auto ceiling = kEdgeCeiling * resolution_;
    if (new_len2 > cur_len2 && new_len2 > ceiling * ceiling) {
      return std::nullopt;  // a lengthening flip may not exceed the hard length ceiling
    }
    Face new_f0{c, x, d};
    Face new_f1{d, y, c};
    auto nn0 = normal(new_f0);
    auto nn1 = normal(new_f1);
    if (!(nn0.norm() > 0.0) || !(nn1.norm() > 0.0)) {
      return std::nullopt;
    }

    // Reject a flip that folds the surface back on itself: a new normal must not oppose the quad
    // normal (the sum of the two old face normals).
    auto n0 = normal(f0);
    auto n1 = normal(f1);
    auto d0 = n0.norm();
    auto d1 = n1.norm();
    if (d0 > 0.0 && d1 > 0.0) {
      Vector3 avg = n0 / d0 + n1 / d1;
      if (nn0.dot(avg) <= 0.0 || nn1.dot(avg) <= 0.0) {
        return std::nullopt;
      }
    }

    Index gi_cx = mesh_.across({c, x}, fi0);
    Index gi_yc = mesh_.across({y, c}, fi0);
    Index gi_xd = mesh_.across({x, d}, fi1);
    Index gi_dy = mesh_.across({d, y}, fi1);

    // Flip whenever the summed bend over the five touched edges strictly drops.
    auto before = bend(f0, f1) + bend_with(f0, gi_cx) + bend_with(f0, gi_yc) +
                  bend_with(f1, gi_xd) + bend_with(f1, gi_dy);
    auto after = bend(new_f0, new_f1) + bend_with(new_f0, gi_cx) + bend_with(new_f1, gi_yc) +
                 bend_with(new_f0, gi_xd) + bend_with(new_f1, gi_dy);
    if (!(after < before - 1e-6)) {
      return std::nullopt;
    }
    auto improve = before - after;

    // Dihedral-dependent length cap, only when the flip lengthens the diagonal: the overshoot past
    // kEdgeFloor * res must be earned by bend reduction. A shortening flip adds no edge longer than
    // the one it removes, so it skips the cap.
    if (new_len2 > cur_len2) {
      auto overshoot = std::sqrt(new_len2) / resolution_ - kEdgeFloor;
      if (overshoot > 0.0) {
        auto rate = kImproveFull / (kEdgeCeiling - kEdgeFloor);
        if (improve < rate * overshoot) {
          return std::nullopt;
        }
      }
    }

    // Reject a flip below min_angle_ unless it improves the worst angle there -- a sliver (a
    // diagonal grazing a collinear vertex, a T-junction the inexact self-int guard misses) is worse
    // than a crease.
    auto after_angle = std::min(min_angle(new_f0), min_angle(new_f1));
    auto before_angle = std::min(min_angle(f0), min_angle(f1));
    if (after_angle < min_angle_ && after_angle < before_angle) {
      return std::nullopt;
    }
    return Flip{fi0, fi1, x, y, c, d, improve};
  }

  Points3 v_;            // vertices in the isotropic frame, where the geometry is measured
  AbstractMesh mesh_;    // working connectivity, edited in place by flips
  Points3 snap_points_;  // snap targets in the isotropic frame
  VecX snap_tols_;       // snapping tolerance per snap point (isotropic-frame distance)
  SpatialGrid snap_grid_;
  SpatialGrid face_grid_;  // face broad-phase for the self-intersection guard
  double resolution_;      // mesh resolution; sets the diagonal length cap
  double min_angle_;       // a flip may not push a triangle's smallest angle below this (0 = off)
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
