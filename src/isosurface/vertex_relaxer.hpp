#pragma once

#include <Eigen/Core>
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

 public:
  VertexRelaxer(const Mesh& mesh, const FieldFunction& field_fn, double isovalue, double resolution,
                const Mat3& aniso, int passes, const geometry::Bbox3& bbox)
      : field_fn_(field_fn),
        isovalue_(isovalue),
        resolution_(resolution),
        aniso_(aniso),
        bbox_(bbox),
        p_(mesh.vertices()),
        mesh_(mesh.faces()),
        face_grid_(resolution, mesh_.num_faces()) {
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      index_face(fi);
    }
    classify_pinned();
    for (auto pass = 0; pass < passes; pass++) {
      auto areas = voronoi_areas();
      Points3 relaxed = p_;
      for (Index vi = 0; vi < relaxed.rows(); vi++) {
        if (!pinned_.at(vi)) {
          relaxed.row(vi) = relaxed_position(vi, areas);
        }
      }
      // Refine the tangentially slid positions back onto the level set. A vertex the projection
      // leaves alone -- already on the level set, as every vertex on a flat region is after a
      // purely tangential slide -- keeps that slide, which try_move below still commits.
      for (const auto& [vi, target] :
           project_to_level_set(field_fn_, isovalue_, relaxed, resolution_, aniso_)) {
        relaxed.row(vi) = target;
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

  // Marks each vertex that must not move: on a sharp edge (a crease or corner) or on the mesh
  // boundary.
  void classify_pinned() {
    pinned_.assign(p_.rows(), false);
    mesh_.for_each_halfedge([&](Halfedge h) {
      auto opp = mesh_.opposite(h);
      if (!opp.is_valid() || bend(face_of(h), face_of(opp)) > kFeatureAngle) {
        pinned_.at(mesh_.from(h)) = true;
        pinned_.at(mesh_.to(h)) = true;
      }
    });
  }

  // The vertex triple of the face incident to halfedge h.
  Face face_of(Halfedge h) const { return mesh_.face(mesh_.face(h)); }

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

  // The tangential move for vertex vi: a damped slide toward the Voronoi-area-weighted average of
  // its one-ring neighbors, with the component along the vertex normal removed so it stays on the
  // surface. Weighting each neighbor by its Voronoi area (areas) targets a uniform vertex
  // distribution. Worked in offsets from vi, so no positions are summed.
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

  // Commits moving vertex vi to target (snapped onto the bbox if within tolerance) iff no incident
  // face inverts or collapses and no new self-intersection results.
  bool try_move(Index vi, const Point3& target) {
    auto new_p = snap_to_bbox(target, bbox_, resolution_);
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

    for (auto fi : mesh_.vertex_faces(vi)) {
      unindex_face(fi);
    }
    p_.row(vi) = new_p;
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
  geometry::Bbox3 bbox_;
  Points3 p_;
  AbstractMesh mesh_;
  FaceGrid face_grid_;
  std::vector<bool> pinned_;
};

}  // namespace polatory::isosurface
