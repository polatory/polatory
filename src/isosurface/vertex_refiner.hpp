#pragma once

#include <Eigen/Core>
#include <array>
#include <jizai/geometry/bbox3d.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/isosurface/field_function.hpp>
#include <jizai/isosurface/mesh.hpp>
#include <jizai/isosurface/types.hpp>
#include <jizai/types.hpp>
#include <utility>
#include <vector>

#include "abstract_mesh.hpp"
#include "face_grid.hpp"
#include "utility.hpp"

namespace jizai::isosurface {

class VertexRefiner {
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using Vector3 = geometry::Vector3;

 public:
  VertexRefiner(const Mesh& mesh, const FieldFunction& field_fn, double isovalue,
                const geometry::Bbox3& bbox, double resolution, const Mat3& aniso)
      : field_fn_(field_fn),
        isovalue_(isovalue),
        bbox_(bbox),
        resolution_(resolution),
        aniso_(aniso),
        p_(mesh.vertices()),
        mesh_(mesh.faces()),
        nv_(p_.rows()),
        face_grid_(resolution, mesh_.num_faces()) {
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      index_face(fi);
    }

    for (const auto& [vi, target] : project()) {
      try_move(vi, target);
    }
  }

  Mesh result() && { return {std::move(p_), std::move(mesh_).take_faces()}; }

 private:
  static constexpr double kMaxMoveRatio = 0.5;

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

  std::vector<std::pair<Index, Point3>> project() const {
    auto d = 1e-3 * resolution_;
    // The vertices of a regular tetrahedron: the rows sum to zero and s^T s = 4 d^2 I.
    Mat<4, 3> s;
    s << Vector3{d, d, d}, Vector3{d, -d, -d}, Vector3{-d, d, -d}, Vector3{-d, -d, d};
    Points3 samples(4 * nv_, 3);
    for (Index i = 0; i < nv_; i++) {
      samples.middleRows(4 * i, 4) = s.rowwise() + p_.row(i);
    }
    VecX v = field_fn_(samples).array() - isovalue_;

    std::vector<std::pair<Index, Point3>> moves;
    auto max_move = kMaxMoveRatio * resolution_;
    auto inv_4dd = 1.0 / (4.0 * d * d);
    for (Index i = 0; i < nv_; i++) {
      auto vs = v.segment(4 * i, 4);
      Vector3 grad = inv_4dd * vs.transpose() * s;
      auto gn2 = grad.squaredNorm();
      if (gn2 == 0.0) {
        continue;
      }
      auto value = 0.25 * vs.sum();
      Vector3 step = (value / gn2) * grad;
      auto a_len = geometry::transform_vector<3>(aniso_, step).norm();
      if (a_len > max_move) {
        step *= max_move / a_len;
      }
      moves.emplace_back(i, Point3(p_.row(i) - step));
    }
    return moves;
  }

  bool try_move(Index vi, const Point3& target) {
    auto new_p = snap_to_bbox(target, bbox_, resolution_);
    for (auto fi : mesh_.vertex_faces(vi)) {
      auto f = mesh_.face(fi);
      auto a = moved_face(fi, vi, new_p);

      // The intersection test below cannot detect a face flipping over.
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
        return triangles_intersect(a.at(0), a.at(1), a.at(2), b.at(0), b.at(1), b.at(2));
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
  geometry::Bbox3 bbox_;
  double resolution_;
  Mat3 aniso_;
  Points3 p_;
  AbstractMesh mesh_;
  Index nv_;
  FaceGrid face_grid_;
};

}  // namespace jizai::isosurface
