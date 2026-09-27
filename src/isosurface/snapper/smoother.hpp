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

class Smoother {
  using Point2 = geometry::Point2;
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using Vector3 = geometry::Vector3;

  static constexpr double kPi = 3.141592653589793;
  static constexpr double kMaxEdgeRatio = 1.3;

  struct Flip {
    Index fi0;
    Index fi1;
    Index x;
    Index y;
    Index c;
    Index d;
    double improve;
    std::array<Index, 4> outer_faces;

    Face new_f0() const { return {c, x, d}; }
    Face new_f1() const { return {d, y, c}; }
  };

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
        enqueue(Edge{mesh_.from(h), mesh_.to(h)});
      }
    });

    std::int64_t flips = 0;
    auto max_flips = 50 * std::max<std::int64_t>(mesh_.num_faces(), 1);  // guards against cycles
    while (!pq.empty()) {
      Edge e = pq.top().e;
      pq.pop();
      auto fl = score(e);  // the queued score may be stale
      if (!fl || !honors_ok(*fl) || self_intersects(*fl)) {
        continue;
      }
      do_flip(*fl);
      if (++flips > max_flips) {
        break;
      }
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

  double bend_with(const Face& a, Index fi) const { return fi < 0 ? 0.0 : bend(a, mesh_.face(fi)); }

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

  bool honored_by(Index i, const Face& f) const {
    return dist2(a_points_.row(i), f) <= snap_tols2_(i);
  }

  bool honors_ok(const Flip& fl) const {
    if (snap_grid_.empty()) {
      return true;
    }
    auto f0 = mesh_.face(fl.fi0);
    auto f1 = mesh_.face(fl.fi1);
    std::array<Face, 6> after{fl.new_f0(), fl.new_f1()};
    auto na = 2;
    for (auto gi : fl.outer_faces) {
      if (gi >= 0) {
        after.at(na++) = mesh_.face(gi);
      }
    }
    auto aps = ap_({fl.x, fl.y, fl.c, fl.d}, kAll);
    Point3 lo = aps.colwise().minCoeff();
    Point3 hi = aps.colwise().maxCoeff();
    bool ok = true;
    snap_grid_.for_each(lo, hi, [&](Index i) {
      auto honored = [&](const auto& f) { return honored_by(i, f); };
      if (!honored(f0) && !honored(f1)) {
        return true;
      }
      if (std::ranges::none_of(after.begin(), after.begin() + na, honored)) {
        ok = false;
        return false;
      }
      return true;
    });
    return ok;
  }

  void index_face(Index fi) { face_grid_.insert(fi, p_(mesh_.face(fi), kAll)); }

  Vector3 normal(const Face& f) const {
    return triangle_normal(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  // Self-intersection is judged on the output positions p_, not ap_.
  bool intersect(const Face& a, const Face& b) const {
    return triangles_intersect(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)), p_.row(b(0)), p_.row(b(1)),
                               p_.row(b(2)));
  }

  std::optional<Flip> score(const Edge& e) const {
    auto h = mesh_.halfedge_of(e.a, e.b);
    auto opp_h = mesh_.opposite(h);
    auto fi0 = mesh_.face(h);
    auto fi1 = mesh_.face(opp_h);
    if (fi0 < 0 || fi1 < 0) {
      return std::nullopt;
    }

    Index x = mesh_.from(h);
    Index y = mesh_.to(h);
    Index c = mesh_.apex(h);
    Index d = mesh_.apex(opp_h);
    if (c == d || mesh_.has_edge({c, d})) {
      return std::nullopt;
    }

    auto f0 = mesh_.face(fi0);
    auto f1 = mesh_.face(fi1);
    Face new_f0{c, x, d};
    Face new_f1{d, y, c};

    Index gi_cx = mesh_.face(mesh_.opposite(mesh_.prev(h)));
    Index gi_yc = mesh_.face(mesh_.opposite(mesh_.next(h)));
    Index gi_xd = mesh_.face(mesh_.opposite(mesh_.next(opp_h)));
    Index gi_dy = mesh_.face(mesh_.opposite(mesh_.prev(opp_h)));

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

    return Flip{fi0, fi1, x, y, c, d, improve, {gi_cx, gi_yc, gi_xd, gi_dy}};
  }

  bool self_intersects(const Flip& fl) {
    return crosses(fl.new_f0(), fl.fi0, fl.fi1) || crosses(fl.new_f1(), fl.fi0, fl.fi1);
  }

  void unindex_face(Index fi) { face_grid_.remove(fi); }

  Points3 p_;
  Points3 ap_;
  AbstractMesh mesh_;
  Points3 a_points_;
  VecX snap_tols2_;
  SpatialGrid snap_grid_;
  FaceGrid face_grid_;
  double max_edge2_;
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
