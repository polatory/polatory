#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <vector>

#include "abstract_mesh.hpp"
#include "face_grid.hpp"
#include "level_set_projection.hpp"
#include "spatial_grid.hpp"
#include "utility.hpp"

namespace polatory::isosurface {

// Improves triangle regularity by tangential relaxation: each free vertex slides toward the
// area-weighted centroid of its star (its incident triangles) within its tangent plane, is
// re-projected onto the level set f = isovalue, and the move is committed only if it introduces no
// fold or self-intersection. Only sharp edges (creases and corners) and the mesh boundary are
// pinned; snap points are left free, since re-projecting onto the level set holds them on the
// features anyway. Geometry and the guard are in the untransformed frame.
class VertexRelaxer {
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using Vector3 = geometry::Vector3;

  static constexpr double kPi = 3.141592653589793;
  static constexpr double kDamping = 0.5;
  static constexpr double kFeatureAngle = 30 * kPi / 180;  // a sharper edge is a crease
  // A star-plane direction counts as constrained when its QEF eigenvalue exceeds this fraction of
  // the largest; below it the direction is free (the crease's along-ridge tangent). Small enough to
  // keep the weak second direction of an obtuse crease, whose two planes are nearly parallel.
  static constexpr double kQefEps = 1e-3;

 public:
  VertexRelaxer(const Mesh& mesh, const FieldFunction& field_fn, double isovalue, double resolution,
                const Mat3& aniso, int passes, const geometry::Bbox3& bbox,
                const geometry::Points3& points, const VecX& tolerances)
      : field_fn_(field_fn),
        isovalue_(isovalue),
        resolution_(resolution),
        aniso_(aniso),
        aniso_inv_(aniso.inverse()),
        bbox_(bbox),
        p_(mesh.vertices()),
        ap_(geometry::transform_points<3>(aniso, mesh.vertices())),
        mesh_(mesh.faces()),
        a_points_(geometry::transform_points<3>(aniso, points)),
        snap_grid_(resolution, points.rows()),
        face_grid_(resolution, mesh_.num_faces()) {
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      index_face(fi);
    }
    VecX tols = tolerances;
    if (tols.size() == 0) {
      tols = VecX::Zero(a_points_.rows());
    }
    snap_grid_.insert_balls(a_points_, tols);
    snap_tols2_ = tols.cwiseAbs2();
    classify();
    for (auto pass = 0; pass < passes; pass++) {
      auto areas = voronoi_areas();
      Points3 relaxed = p_;
      for (Index vi = 0; vi < relaxed.rows(); vi++) {
        if (!pinned_.at(vi)) {
          relaxed.row(vi) =
              feature_.at(vi) ? feature_position(vi, areas) : relaxed_position(vi, areas);
        }
      }
      // Re-project the free (non-feature) vertices onto the level set. A feature vertex is held on
      // the supporting planes of its star instead, off which the level set would pull it.
      for (const auto& [vi, target] :
           project_to_level_set(field_fn_, isovalue_, relaxed, resolution_, aniso_)) {
        if (!feature_.at(vi)) {
          relaxed.row(vi) = target;
        }
      }
      for (Index vi = 0; vi < relaxed.rows(); vi++) {
        if (!pinned_.at(vi)) {
          try_move(vi, relaxed.row(vi));
        }
      }
    }
  }

  Mesh result() && { return {std::move(p_), std::move(mesh_).take_faces()}; }

 private:
  double bend(const Face& a, const Face& b) const {
    Vector3 na = triangle_normal(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)));
    Vector3 nb = triangle_normal(p_.row(b(0)), p_.row(b(1)), p_.row(b(2)));
    auto da = na.norm();
    auto db = nb.norm();
    if (!(da > 0.0) || !(db > 0.0)) {
      return kPi;
    }
    return std::acos(std::clamp(na.dot(nb) / (da * db), -1.0, 1.0));
  }

  // The vertex triple of the face incident to halfedge h.
  Face face_of(Halfedge h) const { return mesh_.face(mesh_.face(h)); }

  // Pins the mesh boundary and marks as a feature vertex any interior vertex on a sharp edge
  // (dihedral > kFeatureAngle). A feature vertex relaxes against the supporting planes of its star,
  // which confine a crease vertex to the ridge line and pin a corner -- the plane count sorts the
  // two out, so the dihedral test only has to separate a feature from a merely curved region.
  void classify() {
    pinned_.assign(p_.rows(), false);
    feature_.assign(p_.rows(), false);
    mesh_.for_each_halfedge([&](Halfedge h) {
      auto opp = mesh_.opposite(h);
      if (!opp.is_valid()) {
        pinned_.at(mesh_.from(h)) = true;
        pinned_.at(mesh_.to(h)) = true;
      } else if (mesh_.from(h) < mesh_.to(h) && bend(face_of(h), face_of(opp)) > kFeatureAngle) {
        feature_.at(mesh_.from(h)) = true;
        feature_.at(mesh_.to(h)) = true;
      }
    });
  }

  void index_face(Index fi) { face_grid_.insert(fi, p_(mesh_.face(fi), kAll)); }

  std::array<Point3, 3> moved_face(Index fi, Index vi, const Point3& new_p) const {
    auto f = mesh_.face(fi);
    std::array<Point3, 3> t;
    for (auto k = 0; k < 3; k++) {
      auto v = f(k);
      t.at(k) = v == vi ? new_p : Point3(p_.row(v));
    }
    return t;
  }

  // The tangential move for a free vertex: a damped slide toward the Voronoi-area-weighted average
  // of its one-ring neighbors (targeting a uniform distribution), with the component along the
  // vertex normal removed so it stays on the surface; the level set refines it afterward.
  Point3 relaxed_position(Index vi, const std::vector<double>& areas) const {
    Vector3 offset = Vector3::Zero();  // area-weighted sum of (neighbor - vi)
    Vector3 normal = Vector3::Zero();
    for (auto fi : mesh_.vertex_faces(vi)) {
      auto f = mesh_.face(fi);
      normal += triangle_normal(p_.row(f(0)), p_.row(f(1)), p_.row(f(2)));
    }
    double total = 0.0;
    for (auto h : mesh_.vertex_outgoing_halfedges(vi)) {
      auto w = mesh_.to(h);
      auto weight = areas.at(w);
      offset += weight * Vector3(p_.row(w) - p_.row(vi));
      total += weight;
    }
    if (!(total > 0.0)) {
      return p_.row(vi);
    }
    Vector3 delta = offset / total;
    auto n2 = normal.squaredNorm();
    if (n2 > 0.0) {
      delta -= (delta.dot(normal) / n2) * normal;  // project out the normal component
    }
    return Point3(p_.row(vi)) + kDamping * delta;
  }

  // The relaxed position for a feature vertex, computed entirely in the aniso-transformed frame (so
  // it respects the anisotropic resolution) and transformed back: the area-weighted centroid pull,
  // then confined to the intersection of the supporting planes of its original star faces -- solved
  // directly as an area-weighted QEF (min over the plane distances) rather than by iterated
  // projection, which crawls when an obtuse crease's two planes are nearly parallel. Truncating the
  // near-zero eigenvalue leaves the along-ridge tangent free: two planes (a crease) confine it to
  // the ridge line, three (a corner) pin it. The level set is not applied to these vertices.
  Point3 feature_position(Index vi, const std::vector<double>& areas) const {
    Vector3 offset = Vector3::Zero();  // area-weighted sum of (neighbor - vi), aniso frame
    double total = 0.0;
    for (auto h : mesh_.vertex_outgoing_halfedges(vi)) {
      auto w = mesh_.to(h);
      auto weight = areas.at(w);
      offset += weight * Vector3(ap_.row(w) - ap_.row(vi));
      total += weight;
    }
    if (!(total > 0.0)) {
      return p_.row(vi);
    }
    Eigen::Vector3d g = (Point3(ap_.row(vi)) + kDamping * (offset / total)).transpose();
    Mat3 a = Mat3::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (auto fi : mesh_.vertex_faces(vi)) {
      auto f = mesh_.face(fi);
      Vector3 nr = triangle_normal(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
      auto w = nr.norm();  // twice the aniso-frame triangle area, the plane's weight
      if (!(w > 0.0)) {
        continue;
      }
      Eigen::Vector3d n = (nr / w).transpose();
      a += w * n * n.transpose();
      b += w * Vector3(ap_.row(f(0))).dot(nr / w) * n;  // w (n . x0) n
    }
    Eigen::SelfAdjointEigenSolver<Mat3> es(a);
    Eigen::Vector3d ev = es.eigenvalues();  // ascending; ev(2) is the largest
    Mat3 v = es.eigenvectors();
    Eigen::Vector3d residual = a * g - b;
    Eigen::Vector3d pos = g;
    for (auto k = 0; k < 3; k++) {
      if (ev(k) > kQefEps * ev(2)) {
        pos -= (v.col(k).dot(residual) / ev(k)) * v.col(k);
      }
    }
    return geometry::transform_point<3>(aniso_inv_, Point3(pos.transpose()));
  }

  // Each vertex's mixed Voronoi area (Meyer et al.): the cotangent-weighted cell area, with the
  // barycentric area/2, area/4 split on an obtuse triangle where the Voronoi cell would fall outside
  // it. Untransformed frame, matching relaxed_position.
  std::vector<double> voronoi_areas() const {
    std::vector<double> areas(p_.rows(), 0.0);
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      auto f = mesh_.face(fi);
      Point3 x0 = p_.row(f(0));
      Point3 x1 = p_.row(f(1));
      Point3 x2 = p_.row(f(2));
      auto cot = [](const Vector3& u, const Vector3& w) {
        auto s = u.cross(w).norm();
        return s > 0.0 ? u.dot(w) / s : 0.0;
      };
      auto d0 = (x1 - x0).dot(x2 - x0);
      auto d1 = (x2 - x1).dot(x0 - x1);
      auto d2 = (x0 - x2).dot(x1 - x2);
      if (d0 < 0.0 || d1 < 0.0 || d2 < 0.0) {
        auto area = 0.5 * (x1 - x0).cross(x2 - x0).norm();
        areas.at(f(0)) += d0 < 0.0 ? area / 2.0 : area / 4.0;
        areas.at(f(1)) += d1 < 0.0 ? area / 2.0 : area / 4.0;
        areas.at(f(2)) += d2 < 0.0 ? area / 2.0 : area / 4.0;
      } else {
        auto l0 = (x2 - x1).squaredNorm();  // opposite x0
        auto l1 = (x0 - x2).squaredNorm();  // opposite x1
        auto l2 = (x1 - x0).squaredNorm();  // opposite x2
        auto cot0 = cot(x1 - x0, x2 - x0);
        auto cot1 = cot(x2 - x1, x0 - x1);
        auto cot2 = cot(x0 - x2, x1 - x2);
        areas.at(f(0)) += (l2 * cot2 + l1 * cot1) / 8.0;
        areas.at(f(1)) += (l2 * cot2 + l0 * cot0) / 8.0;
        areas.at(f(2)) += (l1 * cot1 + l0 * cot0) / 8.0;
      }
    }
    return areas;
  }

  double dist2(const Point3& p, const Face& f) const {
    return point_triangle_dist2(p, ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  // Whether snap point i lies within its tolerance of face f (aniso frame).
  bool honored_by(Index i, const Face& f) const {
    return dist2(a_points_.row(i), f) <= snap_tols2_(i);
  }

  // honored_by for face f with vertex vi moved to a_new_p (its aniso position after the move).
  bool honored_by_moved(Index i, const Face& f, Index vi, const Point3& a_new_p) const {
    Point3 v0 = f(0) == vi ? a_new_p : Point3(ap_.row(f(0)));
    Point3 v1 = f(1) == vi ? a_new_p : Point3(ap_.row(f(1)));
    Point3 v2 = f(2) == vi ? a_new_p : Point3(ap_.row(f(2)));
    return point_triangle_dist2(a_points_.row(i), v0, v1, v2) <= snap_tols2_(i);
  }

  // Whether moving vi (to new_p, aniso a_new_p) leaves every snap point it currently honors still
  // honored: a point held by one of vi's faces must, after the move, be held by a moved face of vi
  // or by an unmoved face nearby -- the edge optimizer's honor guard, for a vertex move.
  bool honors_ok(Index vi, const Point3& new_p, const Point3& a_new_p) const {
    if (snap_grid_.empty()) {
      return true;
    }
    std::vector<Index> star;
    for (auto fi : mesh_.vertex_faces(vi)) {
      star.push_back(fi);
    }
    Point3 alo = a_new_p;
    Point3 ahi = a_new_p;
    Point3 ulo = new_p;
    Point3 uhi = new_p;
    for (auto fi : star) {
      for (auto x : mesh_.face(fi)) {
        Point3 ax = x == vi ? a_new_p : Point3(ap_.row(x));
        alo = alo.cwiseMin(ax);
        ahi = ahi.cwiseMax(ax);
        Point3 ux = x == vi ? new_p : Point3(p_.row(x));
        ulo = ulo.cwiseMin(ux);
        uhi = uhi.cwiseMax(ux);
      }
    }
    bool ok = true;
    snap_grid_.for_each(alo, ahi, [&](Index i) {
      auto face_of = [&](Index fi) { return mesh_.face(fi); };
      if (std::ranges::none_of(star, [&](Index fi) { return honored_by(i, face_of(fi)); })) {
        return true;  // not held by a face of vi; this move cannot dishonor it
      }
      if (std::ranges::any_of(star,
                              [&](Index fi) { return honored_by_moved(i, face_of(fi), vi, a_new_p); }) ||
          face_grid_.any_of(ulo, uhi, [&](Index fj) {
            return std::ranges::find(star, fj) == star.end() && honored_by(i, face_of(fj));
          })) {
        return true;
      }
      ok = false;
      return false;  // dishonored
    });
    return ok;
  }

  // Commits moving vertex vi to target (snapped onto the bbox if within tolerance) iff no incident
  // face inverts or collapses, no new self-intersection results, and no honored snap point is
  // dishonored.
  bool try_move(Index vi, const Point3& target) {
    auto new_p = snap_to_bbox(target, bbox_, resolution_);
    Point3 a_new_p = geometry::transform_point<3>(aniso_, new_p);
    for (auto fi : mesh_.vertex_faces(vi)) {
      auto f = mesh_.face(fi);
      auto a = moved_face(fi, vi, new_p);

      // Reject a move that inverts or collapses an incident face -- a fold across its opposite
      // edge, which the pierce test below cannot see (it skips edge-adjacent pairs). n_old.n_new <=
      // 0 covers both the flip (normal reverses) and the exact collapse (n_new = 0); a sliver that
      // keeps its orientation still passes.
      Vector3 n_old = triangle_normal(p_.row(f(0)), p_.row(f(1)), p_.row(f(2)));
      Vector3 n_new = triangle_normal(a.at(0), a.at(1), a.at(2));
      if (n_old.dot(n_new) <= 0.0) {
        return false;
      }

      Point3 lo = a.at(0).cwiseMin(a.at(1)).cwiseMin(a.at(2));
      Point3 hi = a.at(0).cwiseMax(a.at(1)).cwiseMax(a.at(2));
      bool hit = face_grid_.any_of(lo, hi, [&](Index fj) {
        if (fj == fi) {
          return false;
        }
        auto b = moved_face(fj, vi, new_p);
        auto shared = num_shared_vertices(f, mesh_.face(fj));
        return triangles_intersect(a.at(0), a.at(1), a.at(2), b.at(0), b.at(1), b.at(2), shared);
      });
      if (hit) {
        return false;
      }
    }

    if (!honors_ok(vi, new_p, a_new_p)) {
      return false;
    }

    for (auto fi : mesh_.vertex_faces(vi)) {
      unindex_face(fi);
    }
    p_.row(vi) = new_p;
    ap_.row(vi) = a_new_p;
    for (auto fi : mesh_.vertex_faces(vi)) {
      index_face(fi);
    }
    return true;
  }

  void unindex_face(Index fi) { face_grid_.remove(fi); }

  const FieldFunction& field_fn_;
  double isovalue_;
  double resolution_;
  Mat3 aniso_;
  Mat3 aniso_inv_;
  geometry::Bbox3 bbox_;
  Points3 p_;
  Points3 ap_;         // p_ in the aniso-transformed frame, where snap distances are measured
  AbstractMesh mesh_;
  Points3 a_points_;   // the snap targets, aniso-transformed
  SpatialGrid snap_grid_;
  FaceGrid face_grid_;
  VecX snap_tols2_;    // squared snapping tolerance per snap point
  std::vector<bool> pinned_;
  std::vector<bool> feature_;
};

}  // namespace polatory::isosurface
