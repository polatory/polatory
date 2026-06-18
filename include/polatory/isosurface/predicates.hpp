#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Inexact 2D geometric predicates shared across isosurface generation (triangulation,
// clipping, mesh-defect detection, smoothing). Each evaluates its defining polynomial in
// plain double precision, so a result within a few ULP of zero is unreliable; the caller
// gates on its own tolerance rather than exact zero wherever that distinction matters.

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

// Whether filled triangles a and b overlap with positive area, by the separating-axis test
// (exact for convex polygons up to round-off). slack is a real distance: a gap or contact
// narrower than slack along a separating axis still reads as separated, so triangles that
// merely touch at a shared vertex or edge are not reported as overlapping. The comparison is
// strict, so the test errs toward reporting an overlap -- the safe side when it guards
// against self-intersection.
inline bool triangles_overlap_2d(const std::array<geometry::Point2, 3>& a,
                                 const std::array<geometry::Point2, 3>& b, double slack) {
  // Whether some edge of s lies on a separating axis that keeps t off s (half the test;
  // the caller runs both orderings, since either triangle may carry the separating edge).
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

}  // namespace polatory::isosurface
