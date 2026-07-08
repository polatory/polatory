#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <queue>
#include <utility>
#include <vector>

#include "../abstract_mesh.hpp"
#include "../face_grid.hpp"
#include "../spatial_grid.hpp"
#include "../utility.hpp"

namespace polatory::isosurface::snapper {

// Post-process smoothing by edge flips: flip an interior edge while it lowers the total bend
// (summed dihedral). A flip changes only its five edges, so the local drop equals the global drop
// and the mesh descends to a local minimum; vertices never move, so snapped points stay vertices. A
// priority queue takes the largest improvement first, re-scoring each popped edge (a nearby flip
// may have staled it). Geometry is in the aniso-transformed frame; the output is untransformed. A
// flip is rejected if its new diagonal overshoots the bend-dependent length cap,
// self-intersects, or pushes the surface beyond a snap tolerance (protecting points honored within
// tolerance with no vertex there).
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
  // A flip may shrink the smaller angle only to this fraction of the old.
  static constexpr double kMinAngleRatio = 0.5;
  // A flip whose five touched edges are folded past this in total (summed dihedral) bypasses the
  // length and min-angle quality caps (a severely folded neighbourhood or cave must go), subject to
  // the usual validity checks.
  static constexpr double kSevereFoldSum = 5 * kPi / 3;  // 300 deg
  // A flip may never create a triangle whose area has collapsed below this fraction of the area of
  // the faces it replaces -- a degenerate face the severe-fold bypass would otherwise admit.
  static constexpr double kDegenerateAreaRatio = 1e-9;

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
           const Mat3& aniso)
      : p_(mesh.vertices()),
        ap_(geometry::transform_points<3>(aniso, mesh.vertices())),
        mesh_(mesh.faces()),
        a_points_(geometry::transform_points<3>(aniso, points)),
        snap_grid_(resolution, points.rows()),
        face_grid_(resolution, mesh_.num_faces()),
        resolution_(resolution) {
    // Grid cell = resolution (a face spans about one cell); it only tunes the broad-phase, and
    // resolution avoids a lone long edge blowing the grid up.
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      index_face(fi);
    }

    VecX tols = tolerances;
    if (tols.size() == 0) {
      tols = VecX::Zero(a_points_.rows());
    }
    snap_grid_.insert_balls(a_points_, tols);
    snap_tols2_ = tols.cwiseAbs2();

    std::priority_queue<Item, std::vector<Item>, ItemLess> pq;
    auto enqueue = [&](const Edge& e) {
      if (auto fl = score(e)) {
        pq.push({e, fl->improve});
      }
    };
    mesh_.for_each_halfedge([&](Halfedge h) {
      if (mesh_.from(h) < mesh_.to(h) && mesh_.opposite(h).is_valid()) {
        enqueue(Edge{mesh_.from(h), mesh_.to(h)});  // the canonical side of each interior edge
      }
    });

    std::int64_t flips = 0;
    auto cap = 50 * std::max<std::int64_t>(mesh_.num_faces(), 1);  // backstop against a float cycle
    while (!pq.empty()) {
      Edge e = pq.top().e;
      pq.pop();
      auto fl = score(e);  // re-score: a nearby flip may have outdated this entry
      if (!fl || creates_degenerate(*fl) || !honors_ok(*fl) || !guard_ok(*fl)) {
        continue;
      }
      do_flip(*fl);
      if (++flips > cap) {
        break;
      }
      reenqueue_around(*fl, enqueue);
    }

    // Second pass: flip toward interior valence 6 for more regular triangles. Vertices never move,
    // so the surface stays put; honors_ok keeps snapped features on the mesh.
    std::priority_queue<Item, std::vector<Item>, ItemLess> vpq;
    auto venqueue = [&](const Edge& e) {
      if (auto fl = score_valence(e)) {
        vpq.push({e, fl->improve});
      }
    };
    mesh_.for_each_halfedge([&](Halfedge h) {
      if (mesh_.from(h) < mesh_.to(h) && mesh_.opposite(h).is_valid()) {
        venqueue(Edge{mesh_.from(h), mesh_.to(h)});
      }
    });
    while (!vpq.empty()) {
      Edge e = vpq.top().e;
      vpq.pop();
      auto fl = score_valence(e);
      if (!fl || creates_degenerate(*fl) || !honors_ok(*fl) || !guard_ok(*fl)) {
        continue;
      }
      do_flip(*fl);
      if (++flips > cap) {
        break;
      }
      reenqueue_around(*fl, venqueue);
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

  // Whether new_f intersects any spatially near face; scanning new_f's own cells suffices (the
  // sibling call covers the other new triangle).
  bool crosses(const Face& new_f, Index fi0, Index fi1) {
    auto ps = p_(new_f, kAll);
    Point3 lo = ps.colwise().minCoeff();
    Point3 hi = ps.colwise().maxCoeff();
    return face_grid_.any_of(lo, hi, [&](Index gi) {
      return gi != fi0 && gi != fi1 && overlaps(new_f, mesh_.face(gi));
    });
  }

  // Whether the flip would create a degenerate triangle -- its area collapsed to almost nothing next
  // to the faces it replaces.
  bool creates_degenerate(const Flip& fl) const {
    auto scale = std::max(normal(mesh_.face(fl.fi0)).norm(), normal(mesh_.face(fl.fi1)).norm());
    return normal(fl.new_f0()).norm() <= kDegenerateAreaRatio * scale ||
           normal(fl.new_f1()).norm() <= kDegenerateAreaRatio * scale;
  }

  // The number of edges incident to v (its outgoing halfedges).
  Index degree(Index v) const {
    auto r = mesh_.vertex_outgoing_halfedges(v);
    return static_cast<Index>(std::distance(r.begin(), r.end()));
  }

  double dist2(const Point3& p, const Face& f) const {
    return point_triangle_dist2(p, ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  void do_flip(const Flip& fl) {
    unindex_face(fl.fi0);
    unindex_face(fl.fi1);
    mesh_.flip({fl.x, fl.y});
    index_face(fl.fi0);
    index_face(fl.fi1);
  }

  // The flippable interior edge e as a Flip (improve unset), with the summed five-edge bend before
  // and after the flip returned by reference; nullopt if e is a boundary or its flipped diagonal is
  // degenerate or already present. Shared by score() and score_valence().
  std::optional<Flip> flip_geometry(const Edge& e, double& before, double& after) const {
    auto h = mesh_.halfedge_of(e.a, e.b);  // canonical (e.a < e.b); traverses x -> y in f0
    auto opp_h = mesh_.opposite(h);
    auto fi0 = mesh_.face(h);
    auto fi1 = mesh_.face(opp_h);
    if (fi0 < 0 || fi1 < 0) {
      return std::nullopt;  // not a present interior edge (a boundary, or a prior flip removed it)
    }

    Index x = mesh_.from(h);
    Index y = mesh_.to(h);
    Index c = mesh_.apex(h);
    Index d = mesh_.apex(opp_h);
    if (c == d || mesh_.has_edge({c, d})) {
      return std::nullopt;  // degenerate, or the flipped diagonal already exists
    }

    auto f0 = mesh_.face(fi0);
    auto f1 = mesh_.face(fi1);
    Face new_f0{c, x, d};
    Face new_f1{d, y, c};

    // The quad's four outer neighbours, across the edges next/prev to the flipped edge.
    Index gi_cx = mesh_.face(mesh_.opposite(mesh_.prev(h)));
    Index gi_yc = mesh_.face(mesh_.opposite(mesh_.next(h)));
    Index gi_xd = mesh_.face(mesh_.opposite(mesh_.next(opp_h)));
    Index gi_dy = mesh_.face(mesh_.opposite(mesh_.prev(opp_h)));

    before = bend(f0, f1) + bend_with(f0, gi_cx) + bend_with(f0, gi_yc) + bend_with(f1, gi_xd) +
             bend_with(f1, gi_dy);
    after = bend(new_f0, new_f1) + bend_with(new_f0, gi_cx) + bend_with(new_f1, gi_yc) +
            bend_with(new_f0, gi_xd) + bend_with(new_f1, gi_dy);

    return Flip{fi0, fi1, x, y, c, d, 0.0};
  }

  // Self-intersection guard: neither new triangle may cross a face in its grid cells -- a
  // broad-phase that catches a diagonal passing over a spatially near but topologically distant
  // sheet.
  bool guard_ok(const Flip& fl) {
    return !crosses(fl.new_f0(), fl.fi0, fl.fi1) && !crosses(fl.new_f1(), fl.fi0, fl.fi1);
  }

  // Whether snap point i lies within its tolerance of face f.
  bool honored_by(Index i, const Face& f) const {
    return dist2(a_points_.row(i), f) <= snap_tols2_(i);
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
    // Each outer edge belongs to one removed face; its neighbour across that face also bends. Each
    // halfedge is directed as its removed face (f0 = x -> y -> c, f1 = y -> x -> d) traverses it.
    auto add_neighbour = [&](Index from, Index to) {
      Index gi = mesh_.face(mesh_.opposite(mesh_.halfedge_of(from, to)));
      if (gi >= 0) {
        after.at(na++) = mesh_.face(gi);
      }
    };
    add_neighbour(fl.c, fl.x);
    add_neighbour(fl.y, fl.c);
    add_neighbour(fl.x, fl.d);
    add_neighbour(fl.d, fl.y);
    auto aps = ap_({fl.x, fl.y, fl.c, fl.d}, kAll);
    Point3 lo = aps.colwise().minCoeff();
    Point3 hi = aps.colwise().maxCoeff();
    bool ok = true;
    snap_grid_.for_each(lo, hi, [&](Index i) {
      auto honored = [&](const auto& f) { return honored_by(i, f); };
      if (!honored(f0) && !honored(f1)) {
        return true;  // not honored by a removed face; the flip cannot dishonor it
      }
      if (std::ranges::none_of(after.begin(), after.begin() + na, honored)) {
        ok = false;
        return false;  // dishonored; stop the walk
      }
      return true;
    });
    return ok;
  }

  // Index a face by the grid cells its current AABB touches.
  void index_face(Index fi) { face_grid_.insert(fi, p_(mesh_.face(fi), kAll)); }

  // Whether every edge incident to v has two faces (v is not on the mesh boundary).
  bool is_interior(Index v) const {
    for (auto h : mesh_.vertex_outgoing_halfedges(v)) {
      if (!mesh_.opposite(h).is_valid()) {
        return false;
      }
    }
    return true;
  }

  double min_angle(const Face& f) const {
    return triangle_min_angle(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  // Whether the flip keeps the smaller of its two new triangles' min angle at least kMinAngleRatio
  // of the old -- forbids trading a bend or valence gain for a much worse sliver.
  bool min_angle_ok(const Flip& fl) const {
    auto after_angle = std::min(min_angle(fl.new_f0()), min_angle(fl.new_f1()));
    auto before_angle = std::min(min_angle(mesh_.face(fl.fi0)), min_angle(mesh_.face(fl.fi1)));
    return after_angle >= kMinAngleRatio * before_angle;
  }

  Vector3 normal(const Face& f) const {
    return triangle_normal(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  // Whether the two faces intersect, via triangles_intersect (the defect finder's edge-pierce test)
  // in the untransformed frame -- the same predicate and frame the finder uses, so the guard
  // rejects exactly what it would flag. Edge-adjacent pairs (shared >= 2) legitimately meet at
  // their shared edge, so they are not treated as intersecting.
  bool overlaps(const Face& a, const Face& b) const {
    auto shared = num_shared_vertices(a, b);
    if (shared >= 2) {
      return false;
    }
    return triangles_intersect(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)), p_.row(b(0)), p_.row(b(1)),
                               p_.row(b(2)), shared);
  }

  // Re-score the edges of the flip's two changed faces and of their outer neighbours (a flip alters
  // only these), pushing each back onto the queue via enqueue.
  template <class Enqueue>
  void reenqueue_around(const Flip& fl, const Enqueue& enqueue) const {
    for (Index fi : {fl.fi0, fl.fi1}) {
      for (auto k = 0; k < 3; k++) {
        auto h = mesh_.halfedge(fi, k);
        enqueue(Edge{mesh_.from(h), mesh_.to(h)});
        Index gi = mesh_.face(mesh_.opposite(h));
        if (gi >= 0 && gi != fl.fi0 && gi != fl.fi1) {
          auto g = mesh_.face(gi);
          for (auto m = 0; m < 3; m++) {
            enqueue({g(m), g((m + 1) % 3)});
          }
        }
      }
    }
  }

  // The candidate flip of edge e if it strictly lowers the summed five-edge bend and passes the
  // length and min-angle caps (bypassed only for a severely folded neighbourhood). The
  // self-intersection guard is left to the caller.
  std::optional<Flip> score(const Edge& e) const {
    double before = 0.0;
    double after = 0.0;
    auto fl = flip_geometry(e, before, after);
    if (!fl) {
      return std::nullopt;
    }
    // The reason to flip: the summed bend over the five touched edges must strictly drop.
    if (!(after < before - 1e-6)) {
      return std::nullopt;
    }
    fl->improve = before - after;

    // The length and min-angle quality caps below, applied unless the neighbourhood is severely
    // folded (five-edge dihedral sum >= kSevereFoldSum -- a cave that must go regardless of
    // triangle shape). Validity, the bend-sum drop above, and honors_ok/guard_ok still gate the
    // flip.
    if (before < kSevereFoldSum) {
      // Length: a lengthening flip may not pass the hard ceiling, and any overshoot past kEdgeFloor
      // * res must be earned by bend reduction. A shortening flip adds no longer edge.
      auto new_len2 = (ap_.row(fl->c) - ap_.row(fl->d)).squaredNorm();
      auto cur_len2 = (ap_.row(fl->x) - ap_.row(fl->y)).squaredNorm();
      if (new_len2 > cur_len2) {
        auto new_len = std::sqrt(new_len2);
        if (new_len > kEdgeCeiling * resolution_) {
          return std::nullopt;
        }

        auto overshoot = new_len / resolution_ - kEdgeFloor;
        auto rate = kImproveFull / (kEdgeCeiling - kEdgeFloor);
        if (overshoot > 0.0 && fl->improve < rate * overshoot) {
          return std::nullopt;
        }
      }

      // Min angle: a flip may not shrink the smaller angle below kMinAngleRatio of the old -- a
      // diagonal grazing a collinear vertex or a T-junction the inexact self-intersection guard
      // misses. A mild thinning to flatten a crease is kept: a sliver beats a crease.
      if (!min_angle_ok(*fl)) {
        return std::nullopt;
      }
    }

    return fl;
  }

  // The candidate flip of a flat edge e toward interior valence 6: both the current edge and the
  // new diagonal must be flat (so the quad is planar and the flip shifts no boundary crease), all
  // four vertices interior, and the summed squared valence deviation must strictly drop. improve =
  // that drop. The self-intersection and honor guards are left to the caller.
  std::optional<Flip> score_valence(const Edge& e) const {
    double before = 0.0;
    double after = 0.0;
    auto fl = flip_geometry(e, before, after);
    if (!fl) {
      return std::nullopt;
    }
    // Valence flips run on any interior quad, curved regions and creases included -- every feature
    // is a snap point, so honors_ok rejects a flip that would dishonor one and the iterated + final
    // snap re-forms it. Regularizing only flat quads left the marching-tetrahedra valence pattern on
    // curved surfaces untouched.
    if (!(is_interior(fl->x) && is_interior(fl->y) && is_interior(fl->c) && is_interior(fl->d))) {
      return std::nullopt;  // interior valence 6 is the target only away from the boundary
    }

    auto dev = [](Index deg) {
      auto k = deg - 6;
      return k * k;
    };
    auto before_dev =
        dev(degree(fl->x)) + dev(degree(fl->y)) + dev(degree(fl->c)) + dev(degree(fl->d));
    auto after_dev = dev(degree(fl->x) - 1) + dev(degree(fl->y) - 1) + dev(degree(fl->c) + 1) +
                     dev(degree(fl->d) + 1);
    if (after_dev >= before_dev) {
      return std::nullopt;
    }
    fl->improve = static_cast<double>(before_dev - after_dev);

    // A valence flip earns no length overshoot, so hold the new diagonal to the always-allowed floor
    // and forbid a worse sliver.
    if ((ap_.row(fl->c) - ap_.row(fl->d)).norm() > kEdgeFloor * resolution_) {
      return std::nullopt;
    }
    if (!min_angle_ok(*fl)) {
      return std::nullopt;
    }
    return fl;
  }

  void unindex_face(Index fi) { face_grid_.remove(fi); }

  Points3 p_;
  Points3 ap_;
  AbstractMesh mesh_;  // working connectivity, edited in place by flips
  Points3 a_points_;   // the snap targets
  VecX snap_tols2_;    // squared snapping tolerance per snap point
  SpatialGrid snap_grid_;
  FaceGrid face_grid_;  // face broad-phase for the self-intersection guard
  double resolution_;   // mesh resolution; sets the diagonal length cap
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
