#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <tuple>

namespace polatory::isosurface {

// Inexact geometric predicates.

// Twice the signed area of triangle (a, b, c); positive iff (a, b, c) is counterclockwise.
// The magnitude is meaningful and scales like length squared.
inline double orient2d(const geometry::Point2& a, const geometry::Point2& b,
                       const geometry::Point2& c) {
  Mat2 m;
  m << a(0) - c(0), a(1) - c(1),  //
      b(0) - c(0), b(1) - c(1);
  return m.determinant();
}

inline double orient3d(const geometry::Point3& a, const geometry::Point3& b,
                       const geometry::Point3& c, const geometry::Point3& d) {
  Mat3 m;
  m << a(0) - d(0), a(1) - d(1), a(2) - d(2), b(0) - d(0), b(1) - d(1), b(2) - d(2), c(0) - d(0),
      c(1) - d(1), c(2) - d(2);
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

inline bool segment3_triangle3_intersect_coplanar(const geometry::Point3& p,
                                                  const geometry::Point3& q,
                                                  const geometry::Point3& a,
                                                  const geometry::Point3& b,
                                                  const geometry::Point3& c) {
  geometry::Vector3 n = (b - a).cross(c - a);
  auto abs_nx = std::abs(n(0));
  auto abs_ny = std::abs(n(1));
  auto abs_nz = std::abs(n(2));

  auto i = -1;
  auto j = -1;
  if (abs_nx >= abs_ny && abs_nx >= abs_nz) {
    std::tie(i, j) = n(0) > 0 ? std::make_tuple(1, 2) : std::make_tuple(2, 1);
  } else if (abs_ny >= abs_nx && abs_ny >= abs_nz) {
    std::tie(i, j) = n(1) > 0 ? std::make_tuple(2, 0) : std::make_tuple(0, 2);
  } else {
    std::tie(i, j) = n(2) > 0 ? std::make_tuple(0, 1) : std::make_tuple(1, 0);
  }

  geometry::Point2 p2(p(i), p(j));
  geometry::Point2 q2(q(i), q(j));
  geometry::Point2 a2(a(i), a(j));
  geometry::Point2 b2(b(i), b(j));
  geometry::Point2 c2(c(i), c(j));

  auto pqa = orient2d(p2, q2, a2);
  auto pqb = orient2d(p2, q2, b2);
  auto pqc = orient2d(p2, q2, c2);

  auto sign = [](double x) -> int { return x > 0.0 ? 1 : x < 0.0 ? -1 : 0; };
  auto make_class = [](int a, int b, int c) constexpr -> int {
    return 9 * (a + 1) + 3 * (b + 1) + c + 1;
  };
  switch (make_class(sign(pqa), sign(pqb), sign(pqc))) {
    case make_class(1, 1, 1):
      return false;

    case make_class(1, 1, -1):
    case make_class(0, 1, -1):
    case make_class(1, 0, -1):
    case make_class(0, 0, -1):
    case make_class(1, 1, 0):
      //    B   A                  A              B                                     B   A
      // P ------- Q   or   P -B----- Q   or   P -----A- Q   or   P -B---A- Q   or   P ---C--- Q
      //      C                    C              C                    C
      return orient2d(p2, c2, a2) >= 0.0 && orient2d(q2, b2, c2) >= 0.0;

    case make_class(1, -1, 1):
    case make_class(0, -1, 1):
    case make_class(1, -1, 0):
    case make_class(0, -1, 0):
    case make_class(1, 0, 1):
      //    A   C                  C              A                                     A   C
      // P ------- Q   or   P -A----- Q   or   P -----C- Q   or   P -A---C- Q   or   P ---B--- Q
      //      B                    B              B                    B
      return orient2d(p2, b2, c2) >= 0.0 && orient2d(q2, a2, b2) >= 0.0;

    case make_class(-1, 1, 1):
    case make_class(-1, 1, 0):
    case make_class(-1, 0, 1):
    case make_class(-1, 0, 0):
    case make_class(0, 1, 1):
      //    C   B                  B              C                                     C   B
      // P ------- Q   or   P -C----- Q   or   P -----B- Q   or   P -C---B- Q   or   P ---A--- Q
      //      A                    A              A                    A
      return orient2d(p2, a2, b2) >= 0.0 && orient2d(q2, c2, a2) >= 0.0;

    case make_class(1, -1, -1):
    // case make_class(1, 0, -1):
    // case make_class(1, -1, 0):
    case make_class(1, 0, 0):
    case make_class(0, -1, -1):
      //      A                    A              A                    A
      // P ------- Q   or   P -B----- Q   or   P -----C- Q   or   P -B---C- Q   or   P ---A--- Q
      //    B   C                  C              B                                     B   C
      return orient2d(p2, c2, a2) >= 0.0 && orient2d(q2, a2, b2) >= 0.0;

    case make_class(-1, 1, -1):
    // case make_class(-1, 1, 0):
    // case make_class(0, 1, -1):
    case make_class(0, 1, 0):
    case make_class(-1, 0, -1):
      //      B                    B              B                    B
      // P ------- Q   or   P -C----- Q   or   P -----A- Q   or   P -C---A- Q   or   P ---B--- Q
      //    C   A                  A              C                                     C   A
      return orient2d(p2, a2, b2) >= 0.0 && orient2d(q2, b2, c2) >= 0.0;

    case make_class(-1, -1, 1):
    // case make_class(0, -1, 1):
    // case make_class(-1, 0, 1):
    case make_class(0, 0, 1):
    case make_class(-1, -1, 0):
      //      C                    C              C                    C
      // P ------- Q   or   P -A----- Q   or   P -----B- Q   or   P -A---B- Q   or   P ---C--- Q
      //    A   B                  B              A                                     A   B
      return orient2d(p2, b2, c2) >= 0.0 && orient2d(q2, c2, a2) >= 0.0;

    case make_class(-1, -1, -1):
      return false;

    default:
      // The segment or the triangle is degenerate.
      // This can happen when the segment and the triangle are not coplanar.
      // Return true to enable further checks.
      return true;
  }
}

inline bool segment3_triangle3_intersect(const geometry::Point3& p, const geometry::Point3& q,
                                         const geometry::Point3& a, const geometry::Point3& b,
                                         const geometry::Point3& c) {
  auto abcp = orient3d(a, b, c, p);
  auto abcq = orient3d(a, b, c, q);

  if ((abcp > 0.0 && abcq > 0.0) || (abcp < 0.0 && abcq < 0.0)) {
    return false;
  }

  // For robustness.
  if (!segment3_triangle3_intersect_coplanar(p, q, a, b, c)) {
    return false;
  }

  if (abcp == 0.0 && abcq == 0.0) {
    return true;
  }

  auto pqab = orient3d(p, q, a, b);
  auto pqbc = orient3d(p, q, b, c);
  auto pqca = orient3d(p, q, c, a);

  return (pqab >= 0.0 && pqbc >= 0.0 && pqca >= 0.0) || (pqab <= 0.0 && pqbc <= 0.0 && pqca <= 0.0);
}

}  // namespace polatory::isosurface
