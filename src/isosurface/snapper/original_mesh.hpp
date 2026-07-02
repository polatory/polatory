#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <format>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface::snapper {

using geometry::Point2;
using geometry::Point3;
using geometry::Points3;
using geometry::Vector3;

// The original mesh plus fixed data the snapper reads but never changes: vertex/edge adjacency and
// a cached per-face 2D frame (vertex 0 at the origin, edge 0->1 on x) onto which project() drops a
// point's normal component.
class OriginalMesh {
 public:
  OriginalMesh(Points3 vertices, Faces faces) : v_(std::move(vertices)), f_(std::move(faces)) {
    vertex_faces_.resize(v_.rows());
    for (Index fi = 0; fi < f_.rows(); fi++) {
      for (auto k = 0; k < 3; k++) {
        vertex_faces_.at(f_(fi, k)).push_back(fi);
        edge_faces_[{f_(fi, k), f_(fi, (k + 1) % 3)}].push_back(fi);
      }
    }
  }

  const std::vector<Index>& edge_faces(const Edge& e) const { return edge_faces_.at(e); }

  const Faces& faces() const { return f_; }

  Index num_vertices() const { return v_.rows(); }

  Point2 project(Index fi, const Point3& p) const {
    const auto& fr = frame(fi);
    Vector3 d = p - fr.origin;
    return Point2{d.dot(fr.e1), d.dot(fr.e2)};
  }

  const std::vector<Index>& vertex_faces(Index v) const { return vertex_faces_.at(v); }

  const Points3& vertices() const { return v_; }

 private:
  struct Frame {
    Point3 origin;
    Vector3 e1;
    Vector3 e2;
  };

  Frame compute(Index fi) const {
    auto f = f_.row(fi);
    Point3 a = v_.row(f(0));
    Point3 b = v_.row(f(1));
    Point3 c = v_.row(f(2));
    Vector3 ab = b - a;
    Vector3 ac = c - a;
    Vector3 n = ab.cross(ac);
    if (!(n.squaredNorm() > 0.0)) {
      throw std::runtime_error(std::format("face {} is degenerate", fi));
    }
    Vector3 e1 = ab.normalized();
    Vector3 e2 = n.cross(ab).normalized();
    return {.origin = a, .e1 = e1, .e2 = e2};
  }

  const Frame& frame(Index fi) const {
    auto it = frames_.find(fi);
    if (it == frames_.end()) {
      it = frames_.emplace(fi, compute(fi)).first;
    }
    return it->second;
  }

  Points3 v_;
  Faces f_;
  std::vector<std::vector<Index>> vertex_faces_;
  std::unordered_map<Edge, std::vector<Index>, EdgeHash> edge_faces_;
  mutable std::unordered_map<Index, Frame> frames_;  // lazy projection cache
};

}  // namespace polatory::isosurface::snapper
