#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <boost/container/static_vector.hpp>
#include <cmath>
#include <jizai/geometry/bbox3d.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/isosurface/predicates.hpp>
#include <jizai/isosurface/types.hpp>
#include <jizai/types.hpp>

namespace jizai::isosurface {

inline geometry::Point3 snap_to_bbox(const geometry::Point3& p, const geometry::Bbox3& bbox,
                                     double resolution) {
  const auto& min = bbox.min();
  const auto& max = bbox.max();
  auto tiny = 1e-10 * resolution;

  geometry::Point3 q = p;
  q = ((q.array() - min.array()).abs() < tiny).select(min, q);
  q = ((q.array() - max.array()).abs() < tiny).select(max, q);
  return q;
}

// From Ericson, Real-Time Collision Detection.
inline double point_triangle_closest(const geometry::Point3& p, const geometry::Point3& a,
                                     const geometry::Point3& b, const geometry::Point3& c,
                                     geometry::Point3& closest) {
  geometry::Vector3 ab = b - a;
  geometry::Vector3 ac = c - a;
  geometry::Vector3 ap = p - a;
  double d1 = ab.dot(ap);
  double d2 = ac.dot(ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    closest = a;
    return ap.squaredNorm();
  }
  geometry::Vector3 bp = p - b;
  double d3 = ab.dot(bp);
  double d4 = ac.dot(bp);
  if (d3 >= 0.0 && d4 <= d3) {
    closest = b;
    return bp.squaredNorm();
  }
  geometry::Vector3 cp = p - c;
  double d5 = ab.dot(cp);
  double d6 = ac.dot(cp);
  if (d6 >= 0.0 && d5 <= d6) {
    closest = c;
    return cp.squaredNorm();
  }
  double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    double v = d1 / (d1 - d3);
    closest = a + v * ab;
    return (ap - v * ab).squaredNorm();
  }
  double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    double w = d2 / (d2 - d6);
    closest = a + w * ac;
    return (ap - w * ac).squaredNorm();
  }
  double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    closest = b + w * (c - b);
    return (p - closest).squaredNorm();
  }
  double denom = 1.0 / (va + vb + vc);
  double v = vb * denom;
  double w = vc * denom;
  closest = a + ab * v + ac * w;
  return (p - closest).squaredNorm();
}

inline double point_triangle_dist2(const geometry::Point3& p, const geometry::Point3& a,
                                   const geometry::Point3& b, const geometry::Point3& c) {
  geometry::Point3 closest;
  return point_triangle_closest(p, a, b, c, closest);
}

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

inline geometry::Vector3 triangle_normal(const geometry::Point3& a, const geometry::Point3& b,
                                         const geometry::Point3& c) {
  return geometry::Vector3((b - a).cross(c - a));
}

// Tests if triangles a and b possibly intersect besides the simplices they share.
inline bool triangles_intersect(const geometry::Point3& a0, const geometry::Point3& a1,
                                const geometry::Point3& a2, const geometry::Point3& b0,
                                const geometry::Point3& b1, const geometry::Point3& b2) {
  std::array<geometry::Point3, 3> a{a0, a1, a2};
  std::array<geometry::Point3, 3> b{b0, b1, b2};

  boost::container::static_vector<Index, 3> a_shared;
  boost::container::static_vector<Index, 3> b_shared;
  for (Index i = 0; i < 3; i++) {
    for (Index j = 0; j < 3; j++) {
      if (a.at(i) == b.at(j)) {
        a_shared.push_back(i);
        b_shared.push_back(j);
      }
    }
  }

  switch (a_shared.size()) {
    case 0:
      return triangle3_triangle3_intersect(a0, a1, a2, b0, b1, b2);
    case 1: {
      Index i = (a_shared.at(0) + 1) % 3;
      Index j = (a_shared.at(0) + 2) % 3;
      Index k = (b_shared.at(0) + 1) % 3;
      Index l = (b_shared.at(0) + 2) % 3;
      return segment3_triangle3_intersect(a.at(i), a.at(j), b0, b1, b2) ||
             segment3_triangle3_intersect(b.at(k), b.at(l), a0, a1, a2);
    }
    case 2: {
      Index a_apex = 3 - a_shared.at(0) - a_shared.at(1);
      Index b_apex = 3 - b_shared.at(0) - b_shared.at(1);
      return folded(a.at(a_shared.at(0)), a.at(a_shared.at(1)), a.at(a_apex), b.at(b_apex));
    }
    default:
      return false;
  }
}

}  // namespace jizai::isosurface
