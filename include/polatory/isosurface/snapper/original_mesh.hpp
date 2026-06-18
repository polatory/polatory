#pragma once

#include <igl/AABB.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <format>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::isosurface::snapper {

using geometry::Point2;
using geometry::Point3;
using geometry::Points3;
using geometry::Vector3;

// The original mesh the snapper operates on, together with everything precomputed from it
// that the snapping reads but never changes: the vertex/edge adjacency, an AABB tree for
// nearest-face and box queries, and a cached orthonormal 2D frame per face for projecting
// points onto the face's plane. Everything here is fixed once constructed.
//
// A face's frame places vertex 0 at the origin with the edge to vertex 1 along the x axis;
// project() drops a point's component along the face normal, so a vertex of the face maps to
// its 2D coordinates in the frame and an off-surface point maps to its orthogonal projection
// onto the face's plane. Frames are computed on first use and cached.
class OriginalMesh {
 public:
  OriginalMesh(Points3 vertices, Faces faces) : v_(std::move(vertices)), f_(std::move(faces)) {
    tree_.init(v_, f_);
    vertex_faces_.resize(v_.rows());
    for (Index fi = 0; fi < f_.rows(); fi++) {
      for (auto k = 0; k < 3; k++) {
        vertex_faces_.at(f_(fi, k)).push_back(fi);
        edge_faces_[{f_(fi, k), f_(fi, (k + 1) % 3)}].push_back(fi);
      }
    }
  }

  const Points3& vertices() const { return v_; }

  const Faces& faces() const { return f_; }

  Index num_vertices() const { return v_.rows(); }

  // The faces incident to a vertex / to an undirected edge.
  const std::vector<Index>& vertex_faces(Index v) const { return vertex_faces_.at(v); }
  const std::vector<Index>& edge_faces(const Edge& e) const { return edge_faces_.at(e); }

  // The squared distance from p to the mesh, returning the nearest face fi and the closest
  // point q on it.
  double nearest_face(const Point3& p, int& fi, Point3& q) const {
    return tree_.squared_distance(v_, f_, p, fi, q);
  }

  // Appends to `out` the faces whose bounding box overlaps `query`, skipping those in
  // `exclude` (the broad phase of the piercing check), by descending the AABB tree.
  void faces_near(const Eigen::AlignedBox3d& query, const std::unordered_set<Index>& exclude,
                  std::unordered_set<Index>& out) const {
    descend(tree_, query, exclude, out);
  }

  // The 2D coordinates of point p in face fi's frame.
  Point2 project(Index fi, const Point3& p) const {
    const auto& fr = frame(fi);
    Vector3 d = p - fr.origin;
    return Point2{d.dot(fr.e1), d.dot(fr.e2)};
  }

 private:
  struct Frame {
    Point3 origin;
    Vector3 e1;
    Vector3 e2;
  };

  const Frame& frame(Index fi) const {
    auto it = frames_.find(fi);
    if (it == frames_.end()) {
      it = frames_.emplace(fi, compute(fi)).first;
    }
    return it->second;
  }

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

  static void descend(const igl::AABB<Points3, 3>& node, const Eigen::AlignedBox3d& query,
                      const std::unordered_set<Index>& exclude, std::unordered_set<Index>& out) {
    if (!node.m_box.intersects(query)) {
      return;
    }
    if (node.m_primitive != -1) {
      auto fj = static_cast<Index>(node.m_primitive);
      if (!exclude.contains(fj)) {
        out.insert(fj);
      }
      return;
    }
    if (node.m_left != nullptr) {
      descend(*node.m_left, query, exclude, out);
    }
    if (node.m_right != nullptr) {
      descend(*node.m_right, query, exclude, out);
    }
  }

  Points3 v_;
  Faces f_;
  igl::AABB<Points3, 3> tree_;
  std::vector<std::vector<Index>> vertex_faces_;
  std::unordered_map<Edge, std::vector<Index>, EdgeHash> edge_faces_;
  mutable std::unordered_map<Index, Frame> frames_;  // lazy projection cache
};

}  // namespace polatory::isosurface::snapper
