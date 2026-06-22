#pragma once

#include <igl/tri_tri_intersect.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Inexact geometric predicates shared across isosurface generation (triangulation, clipping,
// mesh-defect detection, snapping, smoothing). The 2D ones evaluate a defining polynomial in
// plain double precision, so a result within a few ULP of zero is unreliable; the caller gates
// on its own tolerance rather than exact zero wherever that distinction matters.

// Twice the signed area of triangle (a, b, c); positive iff (a, b, c) is counterclockwise.
// The magnitude is meaningful and scales like length squared.
inline double orient2d(const geometry::Point2& a, const geometry::Point2& b,
                       const geometry::Point2& c) {
  Mat2 m;
  m << a(0) - c(0), a(1) - c(1),  //
      b(0) - c(0), b(1) - c(1);
  return m.determinant();
}

// Positive iff d lies inside the circumcircle of (a, b, c) when (a, b, c) is counterclockwise
// (the sign flips with the winding). The magnitude scales like length to the fourth.
inline double incircle(const geometry::Point2& a, const geometry::Point2& b,
                       const geometry::Point2& c, const geometry::Point2& d) {
  auto m00 = a(0) - d(0);
  auto m01 = a(1) - d(1);
  auto m02 = m00 * m00 + m01 * m01;
  auto m10 = b(0) - d(0);
  auto m11 = b(1) - d(1);
  auto m12 = m10 * m10 + m11 * m11;
  auto m20 = c(0) - d(0);
  auto m21 = c(1) - d(1);
  auto m22 = m20 * m20 + m21 * m21;
  Mat3 m;
  m << m00, m01, m02,  //
      m10, m11, m12,   //
      m20, m21, m22;
  return m.determinant();
}

// Whether filled triangles a and b overlap with positive area (separating-axis test). slack is
// a real distance: contact narrower than slack along an axis reads as separated, so a bare
// shared-vertex or shared-edge touch is not an overlap. The strict comparison errs toward
// reporting an overlap -- the safe side for a self-intersection guard.
inline bool triangles_overlap_2d(const std::array<geometry::Point2, 3>& a,
                                 const std::array<geometry::Point2, 3>& b, double slack) {
  // Whether an edge of s separates s from t (half the test; run for both orderings).
  auto separated = [slack](const std::array<geometry::Point2, 3>& s,
                           const std::array<geometry::Point2, 3>& t) {
    for (auto e = 0; e < 3; e++) {
      geometry::Vector2 edge = s.at((e + 1) % 3) - s.at(e);
      geometry::Vector2 axis(-edge.y(), edge.x());
      auto len = axis.norm();
      if (!(len > 0.0)) {
        continue;
      }
      // Normalized so slack is a real distance independent of the edge length.
      axis /= len;
      auto smin = std::numeric_limits<double>::infinity();
      auto smax = -smin;
      auto tmin = smin;
      auto tmax = smax;
      for (auto k = 0; k < 3; k++) {
        auto sp = axis.dot(s.at(k));
        auto tp = axis.dot(t.at(k));
        smin = std::min(smin, sp);
        smax = std::max(smax, sp);
        tmin = std::min(tmin, tp);
        tmax = std::max(tmax, tp);
      }
      if (smax < tmin + slack || tmax < smin + slack) {
        return true;
      }
    }
    return false;
  };
  return !separated(a, b) && !separated(b, a);
}

// Whether triangles a and b overlap with positive measure, discounting contact narrower than tol
// (so a bare shared vertex or edge is not an overlap). Two regimes, split by how parallel the
// faces are, because each has a configuration the other mishandles:
//  - Near-parallel: the 3D segment test reports the grazing line of contact between two nearly
//    coplanar faces that merely tile (e.g. a fan around a shared vertex) as a crossing. Instead
//    require b to straddle a's plane and their footprints to overlap in it (triangles_overlap_2d).
//  - Transversal: project-and-separate is invalid, so use the 3D segment test on triangles shrunk
//    toward their centroids, so a bare touch shrinks below tol while a real overlap survives.
// The caller decides by shared-vertex count whether a pair is even worth testing.
inline bool triangles_overlap_3d(const geometry::Point3& a0, const geometry::Point3& a1,
                                 const geometry::Point3& a2, const geometry::Point3& b0,
                                 const geometry::Point3& b1, const geometry::Point3& b2,
                                 double tol) {
  geometry::Vector3 na = (a1 - a0).cross(a2 - a0);
  geometry::Vector3 nb = (b1 - b0).cross(b2 - b0);
  auto la = na.norm();
  auto lb = nb.norm();
  if (!(la > 0.0) || !(lb > 0.0)) {
    return false;
  }
  if (std::abs(na.dot(nb)) >= 0.9 * la * lb) {
    geometry::Vector3 n = na / la;
    auto dmin = std::numeric_limits<double>::infinity();
    auto dmax = -dmin;
    for (const geometry::Point3& p : {b0, b1, b2}) {
      auto d = n.dot(p - a0);
      dmin = std::min(dmin, d);
      dmax = std::max(dmax, d);
    }
    // b overlaps a only if it meets a's plane with positive measure: lying in it (coplanar) or
    // crossing it (points beyond tol on both sides). A b that stays on one side -- even one merely
    // touching the plane at a vertex it shares with a while tilting away (a fan of faces around that
    // vertex) -- does not overlap a, though its projection onto the plane would falsely suggest it.
    bool coplanar = dmin >= -tol && dmax <= tol;
    bool crosses = dmin < -tol && dmax > tol;
    if (!coplanar && !crosses) {
      return false;
    }
    geometry::Vector3 u = (a1 - a0).normalized();
    geometry::Vector3 w = n.cross(u);
    auto proj = [&](const geometry::Point3& p) {
      geometry::Vector3 e = p - a0;
      return geometry::Point2(e.dot(u), e.dot(w));
    };
    std::array<geometry::Point2, 3> pa{proj(a0), proj(a1), proj(a2)};
    std::array<geometry::Point2, 3> pb{proj(b0), proj(b1), proj(b2)};
    return triangles_overlap_2d(pa, pb, tol);
  }
  auto shrunk = [](const geometry::Point3& p0, const geometry::Point3& p1,
                   const geometry::Point3& p2) {
    geometry::Point3 g = p0 + ((p1 - p0) + (p2 - p0)) / 3.0;
    constexpr auto s = 1e-6;
    return std::array<geometry::Point3, 3>{p0 + s * (g - p0), p1 + s * (g - p1), p2 + s * (g - p2)};
  };
  auto sa = shrunk(a0, a1, a2);
  auto sb = shrunk(b0, b1, b2);
  bool coplanar = false;
  geometry::Point3 s;
  geometry::Point3 t;
  if (!igl::tri_tri_intersection_test_3d(sa[0], sa[1], sa[2], sb[0], sb[1], sb[2], coplanar, s, t)) {
    return false;
  }
  return !coplanar && (t - s).norm() > tol;
}

// Whether disjoint triangles a and b cross, by the robust exact 3D test. Unlike the shrink-based
// transversal branch of triangles_overlap_3d it has no false negatives, but it reports even a bare
// touch as a crossing, so it is only for a pair sharing no vertex -- a shared vertex or edge must
// go to triangles_overlap_3d instead. Bbox-rejects first.
inline bool triangles_cross_3d(const geometry::Point3& a0, const geometry::Point3& a1,
                               const geometry::Point3& a2, const geometry::Point3& b0,
                               const geometry::Point3& b1, const geometry::Point3& b2) {
  Eigen::Vector3d aa0 = a0.transpose();
  Eigen::Vector3d aa1 = a1.transpose();
  Eigen::Vector3d aa2 = a2.transpose();
  Eigen::Vector3d bb0 = b0.transpose();
  Eigen::Vector3d bb1 = b1.transpose();
  Eigen::Vector3d bb2 = b2.transpose();
  Eigen::Vector3d amin = aa0.cwiseMin(aa1).cwiseMin(aa2);
  Eigen::Vector3d amax = aa0.cwiseMax(aa1).cwiseMax(aa2);
  Eigen::Vector3d bmin = bb0.cwiseMin(bb1).cwiseMin(bb2);
  Eigen::Vector3d bmax = bb0.cwiseMax(bb1).cwiseMax(bb2);
  if ((amax.array() < bmin.array()).any() || (bmax.array() < amin.array()).any()) {
    return false;
  }
  return igl::tri_tri_overlap_test_3d(aa0, aa1, aa2, bb0, bb1, bb2);
}

// The squared distance from p to triangle (a, b, c) (its closest point), by Ericson's region test.
inline double point_triangle_dist2(const geometry::Point3& p, const geometry::Point3& a,
                                   const geometry::Point3& b, const geometry::Point3& c) {
  geometry::Vector3 ab = b - a;
  geometry::Vector3 ac = c - a;
  geometry::Vector3 ap = p - a;
  double d1 = ab.dot(ap);
  double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return ap.squaredNorm();
  }
  geometry::Vector3 bp = p - b;
  double d3 = ab.dot(bp);
  double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return bp.squaredNorm();
  }
  geometry::Vector3 cp = p - c;
  double d5 = ab.dot(cp);
  double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return cp.squaredNorm();
  }
  double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    double v = d1 / (d1 - d3);
    return (ap - v * ab).squaredNorm();
  }
  double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    double w = d2 / (d2 - d6);
    return (ap - w * ac).squaredNorm();
  }
  double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return (p - (b + w * (c - b))).squaredNorm();
  }
  double denom = 1.0 / (va + vb + vc);
  double v = vb * denom;
  double w = vc * denom;
  return (p - (a + ab * v + ac * w)).squaredNorm();
}

// The smallest interior angle of triangle (a, b, c), in radians (a sliver reads near 0).
inline double triangle_min_angle(const geometry::Point3& a, const geometry::Point3& b,
                                 const geometry::Point3& c) {
  auto angle = [](const geometry::Vector3& u, const geometry::Vector3& w) {
    auto lu = u.norm();
    auto lw = w.norm();
    if (!(lu > 0.0) || !(lw > 0.0)) {
      return 0.0;
    }
    return std::acos(std::clamp(u.dot(w) / (lu * lw), -1.0, 1.0));
  };
  return std::min({angle(b - a, c - a), angle(a - b, c - b), angle(a - c, b - c)});
}

inline int num_shared_vertices(const Face& a, const Face& b) {
  int n = 0;
  for (auto i = 0; i < 3; i++) {
    for (auto j = 0; j < 3; j++) {
      if (a(i) == b(j)) {
        n++;
      }
    }
  }
  return n;
}

// Whether triangles a and b intersect with positive measure, given how many vertices they share
// (so a bare shared-vertex/edge touch of adjacent faces does not count, and shared == 3 is the same
// face). A disjoint pair gets the robust exact test; a vertex/edge-sharing pair gets the crease-
// aware test, which tells a true fold from the bare touch of a sharp crease. See triangles_cross_3d
// and triangles_overlap_3d.
inline bool triangles_intersect(const geometry::Point3& a0, const geometry::Point3& a1,
                                const geometry::Point3& a2, const geometry::Point3& b0,
                                const geometry::Point3& b1, const geometry::Point3& b2, int shared) {
  if (shared >= 3) {
    return false;
  }
  if (shared == 0) {
    return triangles_cross_3d(a0, a1, a2, b0, b1, b2);
  }
  auto tol = 1e-6 * std::max((a1 - a0).norm(), (b1 - b0).norm());
  return triangles_overlap_3d(a0, a1, a2, b0, b1, b2, tol);
}

}  // namespace polatory::isosurface
