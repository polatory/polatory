#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/predicates.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Snaps any coordinate of p within 1e-10 * resolution of a bbox face exactly onto it.
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

// Hashes a point by its exact coordinates, for deduplicating or matching coincident positions.
struct PointHash {
  std::size_t operator()(const geometry::Point3& p) const noexcept {
    std::hash<double> h;
    return h(p.x()) ^ (h(p.y()) << 1) ^ (h(p.z()) << 2);
  }
};

// Ericson's closest-point region test; sets closest to the nearest point of triangle (a, b, c).
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

// The unnormalized normal of triangle (a, b, c); its length is twice the triangle's area.
inline geometry::Vector3 triangle_normal(const geometry::Point3& a, const geometry::Point3& b,
                                         const geometry::Point3& c) {
  return geometry::Vector3((b - a).cross(c - a));
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

// Whether triangles a, b intersect, by the defect finder's segment-triangle (edge-pierce) test: an
// edge of one piercing the other, skipping any edge that runs through a vertex shared with the
// other triangle (it only bare-touches there). That skip alone yields the three cases -- 0 shared:
// all six edges; 1 shared: the two opposite edges; 2 or 3 shared: every edge is skipped, so no
// overlap (an edge-adjacent fold, should one exist, is the caller's to catch).
inline bool triangles_intersect(const geometry::Point3& a0, const geometry::Point3& a1,
                                const geometry::Point3& a2, const geometry::Point3& b0,
                                const geometry::Point3& b1, const geometry::Point3& b2,
                                int shared) {
  if (shared >= 2) {
    return false;
  }

  geometry::Point3 alo = a0.cwiseMin(a1).cwiseMin(a2);
  geometry::Point3 ahi = a0.cwiseMax(a1).cwiseMax(a2);
  geometry::Point3 blo = b0.cwiseMin(b1).cwiseMin(b2);
  geometry::Point3 bhi = b0.cwiseMax(b1).cwiseMax(b2);
  if ((ahi.array() < blo.array()).any() || (bhi.array() < alo.array()).any()) {
    return false;
  }

  std::array<geometry::Point3, 3> a{a0, a1, a2};
  std::array<geometry::Point3, 3> b{b0, b1, b2};
  auto in = [](const geometry::Point3& p, const std::array<geometry::Point3, 3>& t) {
    return p == t.at(0) || p == t.at(1) || p == t.at(2);
  };
  auto pierces = [&](const std::array<geometry::Point3, 3>& f,
                     const std::array<geometry::Point3, 3>& g) {
    for (auto k = 0; k < 3; k++) {
      if (in(f.at(k), g) || in(f.at((k + 1) % 3), g)) {
        continue;
      }
      if (segment3_triangle3_intersect(f.at(k), f.at((k + 1) % 3), g.at(0), g.at(1), g.at(2))) {
        return true;
      }
    }
    return false;
  };
  return pierces(a, b) || pierces(b, a);
}

}  // namespace polatory::isosurface
