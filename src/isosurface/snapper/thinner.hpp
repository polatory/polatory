#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <vector>

#include "abstract_mesh.hpp"
#include "spatial_grid.hpp"
#include "utility.hpp"

namespace polatory::isosurface::snapper {

using geometry::Points3;

// Cross-pass thinning by guarded edge collapse: collapses a snapped vertex that a later pass left
// redundant (collinear between neighbours) onto a neighbour, reaching vertices from any pass that
// the insert-only thinning could not. A collapse is kept only if the dropped point stays within its
// tolerance of the new surface and the mesh stays manifold, unflipped, and self-intersection-free;
// only snapped vertices collapse, so the base lattice is untouched. Geometry is in the
// aniso-transformed frame; the output is untransformed.
class Thinner {
  using Point3 = geometry::Point3;
  using Vector3 = geometry::Vector3;
  static constexpr double kMaxEdgeRatio =
      1.5;  // a collapse may not make an edge longer than this * res

 public:
  Thinner(const Mesh& mesh, const Points3& points, const VecX& tolerances, double resolution,
          const Mat3& aniso)
      : p_(mesh.vertices()),
        ap_(geometry::transform_points<3>(aniso, mesh.vertices())),
        mesh_(mesh.faces()),
        a_points_(geometry::transform_points<3>(aniso, points)),
        snap_grid_(resolution, points.rows()),
        face_grid_(resolution, mesh.faces().rows()),
        max_edge2_(kMaxEdgeRatio * resolution * (kMaxEdgeRatio * resolution)) {
    VecX tols = tolerances;
    if (tols.size() == 0) {
      tols = VecX::Zero(a_points_.rows());
    }
    snap_grid_.insert_balls(a_points_, tols);
    snap_tols2_ = tols.cwiseAbs2();

    // Mark each vertex that coincides with a snap point (snapped vertices are emitted exactly
    // there); only these may collapse, so the base lattice stays put.
    boost::unordered_flat_set<Point3, PointHash> snap_positions;
    snap_positions.reserve(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      snap_positions.insert(points.row(i));
    }
    snapped_.assign(p_.rows(), false);
    for (Index v = 0; v < p_.rows(); v++) {
      snapped_.at(v) = snap_positions.contains(p_.row(v));
    }

    // Greedy collapse to a fixpoint: a collapse can make a neighbour collapsible (a chain of
    // collinear points thins end to end).
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      insert_face(fi);
    }
    bool any = true;
    while (any) {
      any = false;
      for (Index v = 0; v < p_.rows(); v++) {
        if (snapped_.at(v) && try_collapse(v)) {
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
                   const boost::unordered_flat_map<Index, std::vector<Index>>& edge_faces,
                   double& dev) {
    // Link condition: v and w may share only the two vertices opposite edge (v, w), else the
    // collapse folds two sheets into a non-manifold edge.
    boost::unordered_flat_set<Index> across;
    for (auto fi : edge_faces.at(w)) {
      for (auto x : mesh_.face(fi)) {
        if (x != v && x != w) {
          across.insert(x);
        }
      }
    }
    for (const auto& [x, fs] : edge_faces) {
      if (x != w && !across.contains(x) && mesh_.has_edge({x, w})) {
        return false;
      }
    }

    // v's kept faces (those not on edge (v, w)), with v retargeted to w.
    std::vector<Face> kept;
    boost::unordered_flat_set<Index> umbrella(inc.begin(), inc.end());
    for (auto fi : inc) {
      auto f = mesh_.face(fi);
      if (on_edge(f, v, w)) {
        continue;  // collapses to a degenerate sliver, dropped
      }
      Face nf = (f.array() == v).select(w, f);
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
      if (r != w && !across.contains(r) && (ap_.row(w) - ap_.row(r)).squaredNorm() > max_edge2_) {
        return false;
      }
    }

    // Distortion (for picking the least-distorting neighbour): the dropped vertex's distance to the
    // new surface.
    dev = std::numeric_limits<double>::infinity();
    for (const auto& nf : kept) {
      dev = std::min(dev, dist2(ap_.row(v), nf));
    }

    // Faces near the collapse: those incident to the kept faces' vertices but outside the umbrella.
    boost::unordered_flat_set<Index> nearby;
    for (const auto& nf : kept) {
      for (auto x : nf) {
        for (auto fi : mesh_.incident(x)) {
          if (!umbrella.contains(fi)) {
            nearby.insert(fi);
          }
        }
      }
    }

    if (!honors_ok(inc, kept, nearby)) {
      return false;
    }

    // No new face may self-intersect another. A collapse only moves the kept faces, so any new
    // overlap involves one of them; a spatial broad-phase catches a kept face pushed onto a
    // spatially near but topologically distant sheet that the one-ring would miss.
    for (const auto& nf : kept) {
      auto aps = p_(nf, kAll);
      Point3 lo = aps.colwise().minCoeff();
      Point3 hi = aps.colwise().maxCoeff();
      bool hit = false;
      face_grid_.for_each(lo, hi, [&](Index fi) {
        if (umbrella.contains(fi) || !intersects(nf, mesh_.face(fi))) {
          return true;
        }
        hit = true;
        return false;
      });
      if (hit) {
        return false;
      }
    }
    return true;
  }

  double dist2(const Point3& p, const Face& f) const {
    return point_triangle_dist2(p, ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  Mesh emit() {
    Faces faces = std::move(mesh_).take_faces();
    std::vector<bool> used(p_.rows(), false);
    for (Index i = 0; i < faces.rows(); i++) {
      for (auto k = 0; k < 3; k++) {
        used.at(faces(i, k)) = true;
      }
    }
    std::vector<Index> vv(p_.rows(), -1);
    Index n = 0;
    for (Index v = 0; v < p_.rows(); v++) {
      if (used.at(v)) {
        vv.at(v) = n++;
      }
    }
    Points3 vertices(n, 3);
    for (Index v = 0; v < p_.rows(); v++) {
      if (used.at(v)) {
        vertices.row(vv.at(v)) = p_.row(v);
      }
    }
    for (Index i = 0; i < faces.rows(); i++) {
      for (auto k = 0; k < 3; k++) {
        faces(i, k) = vv.at(faces(i, k));
      }
    }
    return {std::move(vertices), std::move(faces)};
  }

  // Whether snap point i lies within its tolerance of face f.
  bool honored_by(Index i, const Face& f) const {
    return dist2(a_points_.row(i), f) <= snap_tols2_(i);
  }

  // Every nearby snap point held by a removed face -- not just the dropped vertex -- must stay
  // within tolerance, else a greedy chain drifts already-dropped points off the surface.
  bool honors_ok(const std::vector<Index>& inc, const std::vector<Face>& kept,
                 const boost::unordered_flat_set<Index>& nearby) const {
    if (snap_grid_.empty()) {
      return true;
    }
    Point3 lo = ap_.row(mesh_.face(inc.front())(0));
    Point3 hi = lo;
    for (auto fi : inc) {
      for (auto x : mesh_.face(fi)) {
        lo = lo.cwiseMin(ap_.row(x));
        hi = hi.cwiseMax(ap_.row(x));
      }
    }
    bool ok = true;
    snap_grid_.for_each(lo, hi, [&](Index i) {
      auto honored = [&](const auto& f) { return honored_by(i, f); };
      auto face_of = [&](Index fi) -> Face { return mesh_.face(fi); };
      if (std::ranges::none_of(inc, honored, face_of)) {
        return true;  // not held by a removed face; the collapse cannot dishonor it
      }
      if (std::ranges::none_of(kept, honored) && std::ranges::none_of(nearby, honored, face_of)) {
        ok = false;
        return false;  // dishonored; stop the walk
      }
      return true;
    });
    return ok;
  }

  void insert_face(Index fi) {
    auto f = mesh_.face(fi);
    auto aps = p_(f, kAll);
    Point3 lo = aps.colwise().minCoeff();
    Point3 hi = aps.colwise().maxCoeff();
    face_grid_.insert(fi, lo, hi);
  }

  // The self-intersection guard runs in the untransformed frame (p_), where defects are judged,
  // matching the defect finder.
  bool intersects(const Face& a, const Face& b) const {
    return triangles_intersect(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)), p_.row(b(0)), p_.row(b(1)),
                               p_.row(b(2)), num_shared_vertices(a, b));
  }

  Vector3 normal(const Face& f) const {
    return Vector3((ap_.row(f(1)) - ap_.row(f(0))).cross(ap_.row(f(2)) - ap_.row(f(0))));
  }

  static bool on_edge(const Face& f, Index a, Index b) {
    bool ha = (f.array() == a).any();
    bool hb = (f.array() == b).any();
    return ha && hb;
  }

  void remove_face(Index fi) {
    auto f = mesh_.face(fi);
    auto aps = p_(f, kAll);
    Point3 lo = aps.colwise().minCoeff();
    Point3 hi = aps.colwise().maxCoeff();
    face_grid_.remove(fi, lo, hi);
  }

  // Collapse v onto its least-distorting admissible neighbour, if any; returns whether it did.
  bool try_collapse(Index v) {
    const auto& inc = mesh_.incident(v);
    if (inc.size() < 3) {
      return false;
    }
    boost::unordered_flat_map<Index, std::vector<Index>>
        edge_faces;  // neighbour w -> faces on edge (v, w)
    for (auto fi : inc) {
      for (auto w : mesh_.face(fi)) {
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
    // Drop the star from the grid before the collapse rewrites it (remove_face reads live
    // geometry), then re-add the retargeted faces.
    for (auto fi : inc) {
      remove_face(fi);
    }
    for (auto fi : mesh_.collapse({v, best}, best)) {
      insert_face(fi);
    }
    return true;
  }

  Points3 p_;
  Points3 ap_;
  AbstractMesh mesh_;  // working connectivity, edited in place by collapses
  Points3 a_points_;   // the snap targets
  VecX snap_tols2_;    // squared snapping tolerance per snap point
  SpatialGrid snap_grid_;
  SpatialGrid face_grid_;
  double max_edge2_{};         // squared cap on a collapsed edge's length
  std::vector<bool> snapped_;  // per vertex; true = a snap point (only these collapse)
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
