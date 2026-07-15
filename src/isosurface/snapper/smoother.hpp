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

// Post-process smoothing by edge flips. Flip an interior edge whenever doing so lowers the total
// bend, meaning the summed dihedral. A flip changes only its five edges, so the local drop equals
// the global drop and the mesh descends to a local minimum. Vertices never move, so snapped points
// stay vertices. A priority queue takes the largest improvement first and re-scores each popped
// edge, since a nearby flip may have staled it. Geometry is in the aniso-transformed frame and the
// output is untransformed. A flip is rejected if its new diagonal exceeds the length cap, which is
// the base lattice's longest edge, if it self-intersects, or if it pushes the surface beyond a snap
// tolerance and would abandon a point that is honored within tolerance with no vertex there.
class Smoother {
  using Point2 = geometry::Point2;
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using Vector3 = geometry::Vector3;

  static constexpr double kPi = 3.141592653589793;
  // kMaxEdgeRatio * res is the base lattice's longest edge (see the flip length cap below).
  static constexpr double kMaxEdgeRatio = 1.3;

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
        max_edge2_(kMaxEdgeRatio * resolution * (kMaxEdgeRatio * resolution)) {
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
      if (!fl || !honors_ok(*fl) || !guard_ok(*fl)) {
        continue;
      }
      do_flip(*fl);
      if (++flips > cap) {
        break;
      }
      // Re-score the edges of the two changed faces and of their neighbours.
      for (Index fi : {fl->fi0, fl->fi1}) {
        for (auto k = 0; k < 3; k++) {
          auto h = mesh_.halfedge(fi, k);
          enqueue(Edge{mesh_.from(h), mesh_.to(h)});
          Index gi = mesh_.face(mesh_.opposite(h));
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

  // Whether new_f intersects any spatially near face; scanning new_f's own cells suffices (the
  // sibling call covers the other new triangle).
  bool crosses(const Face& new_f, Index fi0, Index fi1) {
    auto ps = p_(new_f, kAll);
    Point3 lo = ps.colwise().minCoeff();
    Point3 hi = ps.colwise().maxCoeff();
    return face_grid_.any_of(lo, hi, [&](Index gi) {
      return gi != fi0 && gi != fi1 && intersect(new_f, mesh_.face(gi));
    });
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

  Vector3 normal(const Face& f) const {
    return triangle_normal(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  // p_ is the untransformed frame, where the defect finder judges.
  bool intersect(const Face& a, const Face& b) const {
    return triangles_intersect(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)), p_.row(b(0)), p_.row(b(1)),
                               p_.row(b(2)));
  }

  // The candidate flip of edge e if admissible (interior, new diagonal absent and non-degenerate,
  // within the length cap, lowers the bend). The cheap checks; the self-intersection guard is left
  // to the caller.
  std::optional<Flip> score(const Edge& e) const {
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

    // The reason to flip: the summed bend over the five touched edges must strictly drop.
    auto before = bend(f0, f1) + bend_with(f0, gi_cx) + bend_with(f0, gi_yc) +
                  bend_with(f1, gi_xd) + bend_with(f1, gi_dy);
    auto after = bend(new_f0, new_f1) + bend_with(new_f0, gi_cx) + bend_with(new_f1, gi_yc) +
                 bend_with(new_f0, gi_xd) + bend_with(new_f1, gi_dy);
    if (!(after < before - 1e-6)) {
      return std::nullopt;
    }

    auto improve = before - after;

    auto cap2 = std::max((ap_.row(x) - ap_.row(y)).squaredNorm(), max_edge2_);
    if ((ap_.row(c) - ap_.row(d)).squaredNorm() > cap2) {
      return std::nullopt;
    }

    return Flip{fi0, fi1, x, y, c, d, improve};
  }

  void unindex_face(Index fi) { face_grid_.remove(fi); }

  Points3 p_;
  Points3 ap_;
  AbstractMesh mesh_;  // working connectivity, edited in place by flips
  Points3 a_points_;   // the snap targets
  VecX snap_tols2_;    // squared snapping tolerance per snap point
  SpatialGrid snap_grid_;
  FaceGrid face_grid_;  // face broad-phase for the self-intersection guard
  double max_edge2_;
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
