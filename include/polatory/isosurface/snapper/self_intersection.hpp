#pragma once

#include <igl/AABB.h>
#include <igl/tri_tri_intersect.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <vector>

namespace polatory::isosurface::snapper {

namespace detail {

inline void collect_overlapping(const igl::AABB<geometry::Points3, 3>& node,
                                const Eigen::AlignedBox3d& query, std::vector<Index>& out) {
  if (!node.m_box.intersects(query)) {
    return;
  }
  if (node.m_primitive != -1) {
    out.push_back(node.m_primitive);
    return;
  }
  if (node.m_left != nullptr) {
    collect_overlapping(*node.m_left, query, out);
  }
  if (node.m_right != nullptr) {
    collect_overlapping(*node.m_right, query, out);
  }
}

}  // namespace detail

// The set of face indices involved in a self-intersection: a pair of faces that share no vertex
// yet meet in 3D (a transversal crossing or a coplanar overlap). An AABB tree provides the broad
// phase; the exact test is igl's triangle-triangle overlap.
inline std::unordered_set<Index> self_intersecting_faces(const geometry::Points3& v,
                                                         const Faces& f) {
  igl::AABB<geometry::Points3, 3> tree;
  tree.init(v, f);

  auto pt = [&](Index i) { return geometry::Point3(v.row(i)); };
  auto shares_vertex = [&](Index i, Index j) {
    for (auto a = 0; a < 3; a++) {
      for (auto b = 0; b < 3; b++) {
        if (f(i, a) == f(j, b)) {
          return true;
        }
      }
    }
    return false;
  };

  std::unordered_set<Index> bad;
  std::vector<Index> cand;
  for (Index i = 0; i < f.rows(); i++) {
    Eigen::AlignedBox3d box;
    for (auto k = 0; k < 3; k++) {
      box.extend(v.row(f(i, k)).transpose());
    }
    cand.clear();
    detail::collect_overlapping(tree, box, cand);
    for (auto j : cand) {
      if (j <= i || shares_vertex(i, j)) {
        continue;
      }
      if (igl::tri_tri_overlap_test_3d(pt(f(i, 0)), pt(f(i, 1)), pt(f(i, 2)), pt(f(j, 0)),
                                       pt(f(j, 1)), pt(f(j, 2)))) {
        bad.insert(i);
        bad.insert(j);
      }
    }
  }
  return bad;
}

}  // namespace polatory::isosurface::snapper
