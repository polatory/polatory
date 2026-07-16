#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Inexact geometric predicates.

inline constexpr double kTinyFactor = 1e-12;

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
  m << a(0) - d(0), a(1) - d(1), a(2) - d(2),  //
      b(0) - d(0), b(1) - d(1), b(2) - d(2),   //
      c(0) - d(0), c(1) - d(1), c(2) - d(2);
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

inline bool segment3_segment3_intersect_1d(const geometry::Point3& a, const geometry::Point3& b,
                                           const geometry::Point3& p, const geometry::Point3& q) {
  geometry::Point3 lo = a.cwiseMin(b).cwiseMin(p).cwiseMin(q);
  geometry::Point3 hi = a.cwiseMax(b).cwiseMax(p).cwiseMax(q);
  Index i = 0;
  (hi - lo).maxCoeff(&i);

  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto ap = a(i) < p(i) ? -1.0 : a(i) == p(i) ? 0.0 : a(i) > p(i) ? 1.0 : nan;
  auto aq = a(i) < q(i) ? -1.0 : a(i) == q(i) ? 0.0 : a(i) > q(i) ? 1.0 : nan;
  auto bp = b(i) < p(i) ? -1.0 : b(i) == p(i) ? 0.0 : b(i) > p(i) ? 1.0 : nan;
  auto bq = b(i) < q(i) ? -1.0 : b(i) == q(i) ? 0.0 : b(i) > q(i) ? 1.0 : nan;

  if (ap < 0.0 && aq < 0.0 && bp < 0.0 && bq < 0.0) {
    return false;
  }

  if (ap > 0.0 && aq > 0.0 && bp > 0.0 && bq > 0.0) {
    return false;
  }

  return true;
}

inline bool segment3_segment3_intersect_2d(const geometry::Point3& a, const geometry::Point3& b,
                                           const geometry::Point3& p, const geometry::Point3& q,
                                           double abp, double abq, double apq, double bpq) {
  auto lo = a.cwiseMin(b).cwiseMin(p).cwiseMin(q);
  auto hi = a.cwiseMax(b).cwiseMax(p).cwiseMax(q);
  auto scale = (hi - lo).norm();
  auto tiny = kTinyFactor * scale * scale;

  if ((abp < -tiny && abq < -tiny) || (abp > tiny && abq > tiny)) {
    return false;
  }

  if ((apq < -tiny && bpq < -tiny) || (apq > tiny && bpq > tiny)) {
    return false;
  }

  return segment3_segment3_intersect_1d(a, b, p, q);
}

inline bool segment3_triangle3_intersect_2d(const geometry::Point3& a, const geometry::Point3& b,
                                            const geometry::Point3& p, const geometry::Point3& q,
                                            const geometry::Point3& r, Index i, Index j,
                                            double scale) {
  auto tiny = kTinyFactor * scale * scale;

  geometry::Point2 a2(a(i), a(j));
  geometry::Point2 b2(b(i), b(j));
  geometry::Point2 p2(p(i), p(j));
  geometry::Point2 q2(q(i), q(j));
  geometry::Point2 r2(r(i), r(j));

  auto apq = orient2d(a2, p2, q2);
  auto aqr = orient2d(a2, q2, r2);
  auto arp = orient2d(a2, r2, p2);
  auto bpq = orient2d(b2, p2, q2);
  auto bqr = orient2d(b2, q2, r2);
  auto brp = orient2d(b2, r2, p2);

  if ((apq <= tiny && aqr <= tiny && arp <= tiny) ||
      (apq >= -tiny && aqr >= -tiny && arp >= -tiny)) {
    return true;
  }

  if ((bpq <= tiny && bqr <= tiny && brp <= tiny) ||
      (bpq >= -tiny && bqr >= -tiny && brp >= -tiny)) {
    return true;
  }

  auto abp = orient2d(a2, b2, p2);
  auto abq = orient2d(a2, b2, q2);
  auto abr = orient2d(a2, b2, r2);

  return segment3_segment3_intersect_2d(a, b, p, q, abp, abq, apq, bpq) ||
         segment3_segment3_intersect_2d(a, b, q, r, abq, abr, aqr, bqr) ||
         segment3_segment3_intersect_2d(a, b, r, p, abr, abp, arp, brp);
}

inline bool segment3_triangle3_intersect(const geometry::Point3& a, const geometry::Point3& b,
                                         const geometry::Point3& p, const geometry::Point3& q,
                                         const geometry::Point3& r, double apqr, double bpqr) {
  geometry::Point3 lo = a.cwiseMin(b).cwiseMin(p).cwiseMin(q).cwiseMin(r);
  geometry::Point3 hi = a.cwiseMax(b).cwiseMax(p).cwiseMax(q).cwiseMax(r);
  auto scale = (hi - lo).norm();
  auto tiny = kTinyFactor * scale * scale * scale;

  if ((apqr < -tiny && bpqr < -tiny) || (apqr > tiny && bpqr > tiny)) {
    return false;
  }

  Index k = 0;
  (hi - lo).minCoeff(&k);
  auto i = (k + 1) % 3;
  auto j = (k + 2) % 3;

  if (!segment3_triangle3_intersect_2d(a, b, p, q, r, i, j, scale)) {
    return false;
  }

  auto abpq = orient3d(a, b, p, q);
  auto abqr = orient3d(a, b, q, r);
  auto abrp = orient3d(a, b, r, p);

  return (abpq <= tiny && abqr <= tiny && abrp <= tiny) ||
         (abpq >= -tiny && abqr >= -tiny && abrp >= -tiny);
}

inline bool segment3_triangle3_intersect(const geometry::Point3& a, const geometry::Point3& b,
                                         const geometry::Point3& p, const geometry::Point3& q,
                                         const geometry::Point3& r) {
  geometry::Point3 ab_lo = a.cwiseMin(b);
  geometry::Point3 ab_hi = a.cwiseMax(b);
  geometry::Point3 pqr_lo = p.cwiseMin(q).cwiseMin(r);
  geometry::Point3 pqr_hi = p.cwiseMax(q).cwiseMax(r);
  if ((ab_hi.array() < pqr_lo.array()).any() || (pqr_hi.array() < ab_lo.array()).any()) {
    return false;
  }

  auto apqr = orient3d(a, p, q, r);
  auto bpqr = orient3d(b, p, q, r);

  return segment3_triangle3_intersect(a, b, p, q, r, apqr, bpqr);
}

inline bool triangle3_triangle3_intersect(const geometry::Point3& a, const geometry::Point3& b,
                                          const geometry::Point3& c, const geometry::Point3& p,
                                          const geometry::Point3& q, const geometry::Point3& r) {
  geometry::Point3 abc_lo = a.cwiseMin(b).cwiseMin(c);
  geometry::Point3 abc_hi = a.cwiseMax(b).cwiseMax(c);
  geometry::Point3 pqr_lo = p.cwiseMin(q).cwiseMin(r);
  geometry::Point3 pqr_hi = p.cwiseMax(q).cwiseMax(r);
  if ((abc_hi.array() < pqr_lo.array()).any() || (pqr_hi.array() < abc_lo.array()).any()) {
    return false;
  }

  auto apqr = orient3d(a, p, q, r);
  auto bpqr = orient3d(b, p, q, r);
  auto cpqr = orient3d(c, p, q, r);
  auto pabc = orient3d(p, a, b, c);
  auto qabc = orient3d(q, a, b, c);
  auto rabc = orient3d(r, a, b, c);

  return segment3_triangle3_intersect(a, b, p, q, r, apqr, bpqr) ||
         segment3_triangle3_intersect(b, c, p, q, r, bpqr, cpqr) ||
         segment3_triangle3_intersect(c, a, p, q, r, cpqr, apqr) ||
         segment3_triangle3_intersect(p, q, a, b, c, pabc, qabc) ||
         segment3_triangle3_intersect(q, r, a, b, c, qabc, rabc) ||
         segment3_triangle3_intersect(r, p, a, b, c, rabc, pabc);
}

inline bool folded_2d(const geometry::Point3& a, const geometry::Point3& b,
                      const geometry::Point3& p, const geometry::Point3& q, Index i, Index j,
                      double scale) {
  auto tiny = kTinyFactor * scale * scale;

  geometry::Point2 a2(a(i), a(j));
  geometry::Point2 b2(b(i), b(j));
  geometry::Point2 p2(p(i), p(j));
  geometry::Point2 q2(q(i), q(j));

  auto abp = orient2d(a2, b2, p2);
  auto abq = orient2d(a2, b2, q2);

  if ((abp < -tiny && abq > tiny) || (abp > tiny && abq < -tiny)) {
    return false;
  }

  return true;
}

// Tests if the triangles (a, b, p) and (a, b, q) are folded over each other.
inline bool folded(const geometry::Point3& a, const geometry::Point3& b, const geometry::Point3& p,
                   const geometry::Point3& q) {
  geometry::Point3 lo = a.cwiseMin(b).cwiseMin(p).cwiseMin(q);
  geometry::Point3 hi = a.cwiseMax(b).cwiseMax(p).cwiseMax(q);
  auto scale = (hi - lo).norm();
  auto tiny = kTinyFactor * scale * scale * scale;

  Index k = 0;
  (hi - lo).minCoeff(&k);
  auto i = (k + 1) % 3;
  auto j = (k + 2) % 3;

  if (!folded_2d(a, b, p, q, i, j, scale)) {
    return false;
  }

  auto abpq = orient3d(a, b, p, q);

  return std::abs(abpq) <= tiny;
}

}  // namespace polatory::isosurface
