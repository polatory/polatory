#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <boost/unordered/unordered_flat_set.hpp>
#include <cstddef>
#include <functional>
#include <jizai/geometry/point3d.hpp>
#include <jizai/isosurface/edge.hpp>
#include <jizai/isosurface/mesh.hpp>
#include <jizai/isosurface/types.hpp>
#include <jizai/types.hpp>
#include <limits>
#include <vector>

#include "../abstract_mesh.hpp"
#include "../face_grid.hpp"
#include "../spatial_grid.hpp"
#include "../utility.hpp"

namespace jizai::isosurface::snapper {

using geometry::Points3;

class Thinner {
  using Point3 = geometry::Point3;
  using Vector3 = geometry::Vector3;

  static constexpr double kMaxEdgeRatio = 1.3;

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

    // Snapped vertices lie exactly at their snap points.
    boost::unordered_flat_set<Point3, PointHash> snap_positions;
    snap_positions.reserve(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      snap_positions.insert(points.row(i));
    }
    snapped_.assign(p_.rows(), false);
    for (Index v = 0; v < p_.rows(); v++) {
      snapped_.at(v) = snap_positions.contains(p_.row(v));
    }

    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      index_face(fi);
    }
    bool collapsed = true;
    while (collapsed) {
      collapsed = false;
      for (Index v = 0; v < p_.rows(); v++) {
        if (snapped_.at(v) && try_collapse(v)) {
          collapsed = true;
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

  bool collapse_ok(Halfedge h, const std::vector<Halfedge>& outgoing, double& dev) {
    auto a = mesh_.from(h);
    auto b = mesh_.to(h);
    if (!snapped_.at(b)) {
      return false;
    }
    auto c = mesh_.apex(h);
    auto d = mesh_.apex(mesh_.opposite(h));

    // The link condition: c and d are the only common neighbors of a and b.
    for (auto hh : outgoing) {
      auto v = mesh_.to(hh);
      if (v != b && v != c && v != d && mesh_.has_edge({v, b})) {
        return false;
      }
    }

    std::vector<Face> kept;
    boost::unordered_flat_set<Index> star;
    for (auto hh : outgoing) {
      auto fi = mesh_.face(hh);
      star.insert(fi);
      auto f = mesh_.face(fi);
      if (on_edge(f, a, b)) {
        continue;
      }
      Face nf = (f.array() == a).select(b, f);
      if (normal(nf).dot(normal(f)) < 0.0) {
        return false;
      }
      kept.push_back(nf);
    }
    if (kept.empty()) {
      return false;
    }

    for (auto hh : outgoing) {
      auto v = mesh_.to(hh);
      if (v != b && v != c && v != d && (ap_.row(b) - ap_.row(v)).squaredNorm() > max_edge2_) {
        return false;
      }
    }

    dev = std::numeric_limits<double>::infinity();
    for (const auto& nf : kept) {
      dev = std::min(dev, dist2(ap_.row(a), nf));
    }

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

    for (const auto& nf : kept) {
      auto ps = p_(nf, kAll);
      Point3 lo = ps.colwise().minCoeff();
      Point3 hi = ps.colwise().maxCoeff();
      if (face_grid_.any_of(lo, hi, [&](Index fi) {
            return !star.contains(fi) && intersect(nf, mesh_.face(fi));
          })) {
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

  bool honored_by(Index i, const Face& f) const {
    return dist2(a_points_.row(i), f) <= snap_tols2_(i);
  }

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
        return true;
      }
      if (std::ranges::none_of(kept, honored) && std::ranges::none_of(nearby, honored, face_of)) {
        ok = false;
        return false;
      }
      return true;
    });
    return ok;
  }

  void index_face(Index fi) { face_grid_.insert(fi, p_(mesh_.face(fi), kAll)); }

  // Self-intersection is judged on the output positions p_, not ap_.
  bool intersect(const Face& a, const Face& b) const {
    return triangles_intersect(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)), p_.row(b(0)), p_.row(b(1)),
                               p_.row(b(2)));
  }

  Vector3 normal(const Face& f) const {
    return triangle_normal(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  static bool on_edge(const Face& f, Index a, Index b) {
    return (f.array() == a).any() && (f.array() == b).any();
  }

  bool try_collapse(Index v) {
    auto range = mesh_.vertex_outgoing_halfedges(v);
    std::vector<Halfedge> outgoing(range.begin(), range.end());  // copy: collapse rewrites it
    if (outgoing.size() < 3) {
      return false;
    }
    if (std::ranges::any_of(outgoing, [&](Halfedge h) { return !mesh_.opposite(h).is_valid(); })) {
      return false;
    }

    Halfedge best{};
    double best_dev = std::numeric_limits<double>::infinity();
    for (auto h : outgoing) {
      double dev = 0.0;
      if (collapse_ok(h, outgoing, dev) && dev < best_dev) {
        best = h;
        best_dev = dev;
      }
    }
    if (!best.is_valid()) {
      return false;
    }
    for (auto h : outgoing) {
      unindex_face(mesh_.face(h));
    }
    for (auto fi : mesh_.collapse(best)) {
      index_face(fi);
    }
    return true;
  }

  void unindex_face(Index fi) { face_grid_.remove(fi); }

  Points3 p_;
  Points3 ap_;
  AbstractMesh mesh_;
  Points3 a_points_;
  VecX snap_tols2_;
  SpatialGrid snap_grid_;
  FaceGrid face_grid_;
  double max_edge2_{};
  std::vector<bool> snapped_;
  Mesh result_;
};

}  // namespace jizai::isosurface::snapper
