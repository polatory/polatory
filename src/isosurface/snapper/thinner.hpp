#pragma once

#include <Eigen/Core>
#include <algorithm>
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
      index_face(fi);
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
  bool collapse_ok(Halfedge h, const std::vector<Halfedge>& hs, double& dev) {
    auto a = mesh_.from(h);  // the dropped vertex
    auto b = mesh_.to(h);    // the kept vertex
    auto c = mesh_.apex(h);
    auto d = mesh_.apex(mesh_.opposite(h));

    // The link condition: Lk(a) \cap Lk(b) =? Lk(a \cup b) = {c, d}.
    for (auto hh : hs) {
      auto v = mesh_.to(hh);
      if (v != b && v != c && v != d && mesh_.has_edge({v, b})) {
        // v \in Lk(a) \cap Lk(b).
        return false;
      }
    }

    // a's kept faces (those not on edge ab), with a retargeted to b.
    std::vector<Face> kept;
    boost::unordered_flat_set<Index> star;
    for (auto hh : hs) {
      auto fi = mesh_.face(hh);
      star.insert(fi);
      auto f = mesh_.face(fi);
      if (on_edge(f, a, b)) {
        continue;  // collapses to a degenerate sliver, dropped
      }
      Face nf = (f.array() == a).select(b, f);
      auto nn = normal(nf);
      if (!(nn.norm() > 0.0)) {
        return false;  // the face would become degenerate
      }
      if (nn.dot(normal(f)) <= 0.0) {
        return false;  // the face would flip
      }
      kept.push_back(nf);
    }
    if (kept.empty()) {
      return false;
    }

    // Cap edge length to keep triangles regular.
    for (auto hh : hs) {
      auto v = mesh_.to(hh);
      if (v != b && v != c && v != d && (ap_.row(b) - ap_.row(v)).squaredNorm() > max_edge2_) {
        return false;
      }
    }

    // Distortion (for picking the least-distorting neighbour): the dropped vertex's distance to the
    // new surface.
    dev = std::numeric_limits<double>::infinity();
    for (const auto& nf : kept) {
      dev = std::min(dev, dist2(ap_.row(a), nf));
    }

    // Faces near the collapse: those incident to the kept faces' vertices but outside the star.
    boost::unordered_flat_set<Index> nearby;
    for (const auto& nf : kept) {
      for (auto v : nf) {
        for (auto fi : mesh_.vertex_faces(v)) {
          if (!star.contains(fi)) {
            nearby.insert(fi);
          }
        }
      }
    }

    if (!honors_ok(star, kept, nearby)) {
      return false;
    }

    // No new face may self-intersect another. A collapse only moves the kept faces, so any new
    // overlap involves one of them; a spatial broad-phase catches a kept face pushed onto a
    // spatially near but topologically distant sheet that the one-ring would miss.
    for (const auto& nf : kept) {
      auto ps = p_(nf, kAll);
      Point3 lo = ps.colwise().minCoeff();
      Point3 hi = ps.colwise().maxCoeff();
      bool hit = false;
      face_grid_.for_each(lo, hi, [&](Index fi) {
        if (star.contains(fi) || !intersects(nf, mesh_.face(fi))) {
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
    Points3 vertices(p_.rows(), 3);
    auto faces = std::move(mesh_).take_faces();
    Index nv = 0;
    std::vector<Index> vv(p_.rows(), -1);
    for (auto f : faces.rowwise()) {
      for (auto k = 0; k < 3; k++) {
        auto v = f(k);
        if (vv.at(v) < 0) {
          vv.at(v) = nv;
          vertices.row(nv) = p_.row(v);
          nv++;
        }
        f(k) = vv.at(v);
      }
    }
    vertices.conservativeResize(nv, Eigen::NoChange);
    return {std::move(vertices), std::move(faces)};
  }

  // Whether snap point i lies within its tolerance of face f.
  bool honored_by(Index i, const Face& f) const {
    return dist2(a_points_.row(i), f) <= snap_tols2_(i);
  }

  // Every nearby snap point held by a removed face -- not just the dropped vertex -- must stay
  // within tolerance, else a greedy chain drifts already-dropped points off the surface.
  bool honors_ok(const boost::unordered_flat_set<Index>& star, const std::vector<Face>& kept,
                 const boost::unordered_flat_set<Index>& nearby) const {
    if (snap_grid_.empty()) {
      return true;
    }
    auto face_of = [&](Index fi) -> Face { return mesh_.face(fi); };
    Point3 lo = ap_.row(face_of(*star.begin())(0));
    Point3 hi = lo;
    for (auto fi : star) {
      for (auto x : face_of(fi)) {
        lo = lo.cwiseMin(ap_.row(x));
        hi = hi.cwiseMax(ap_.row(x));
      }
    }
    bool ok = true;
    snap_grid_.for_each(lo, hi, [&](Index i) {
      auto honored = [&](const auto& f) { return honored_by(i, f); };
      if (std::ranges::none_of(star, honored, face_of)) {
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

  void index_face(Index fi) {
    auto f = mesh_.face(fi);
    auto ps = p_(f, kAll);
    Point3 lo = ps.colwise().minCoeff();
    Point3 hi = ps.colwise().maxCoeff();
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
    return (f.array() == a).any() && (f.array() == b).any();
  }

  // Collapse v onto its least-distorting admissible neighbour, if any; returns whether it did.
  bool try_collapse(Index v) {
    auto out = mesh_.vertex_outgoing_halfedges(v);
    std::vector<Halfedge> hs(out.begin(), out.end());  // copy: collapse rewrites the adjacency
    if (hs.size() < 3) {
      return false;
    }
    // hs holds v's outgoing halfedges v -> w. v must be an interior manifold vertex: each one's
    // opposite (w -> v) must also have a face.
    if (std::ranges::any_of(hs, [&](Halfedge h) { return !mesh_.opposite(h).is_valid(); })) {
      return false;
    }

    Halfedge best{};  // the chosen collapse v -> w
    double best_dev = std::numeric_limits<double>::infinity();
    for (auto h : hs) {
      double dev = 0.0;
      if (collapse_ok(h, hs, dev) && dev < best_dev) {
        best = h;
        best_dev = dev;
      }
    }
    if (!best.is_valid()) {
      return false;
    }
    // Drop the star from the grid before the collapse rewrites it (unindex_face reads live
    // geometry), then re-add the retargeted faces.
    for (auto h : hs) {
      unindex_face(mesh_.face(h));
    }
    for (auto fi : mesh_.collapse(best)) {  // drop best.from onto best.to
      index_face(fi);
    }
    return true;
  }

  void unindex_face(Index fi) {
    auto f = mesh_.face(fi);
    auto ps = p_(f, kAll);
    Point3 lo = ps.colwise().minCoeff();
    Point3 hi = ps.colwise().maxCoeff();
    face_grid_.remove(fi, lo, hi);
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
