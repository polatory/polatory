#pragma once

#include <igl/tri_tri_intersect.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface {

// Post-process smoothing by edge flips: repeatedly flip an interior edge when doing so lowers the
// worst bend (the largest angle between adjacent faces' normals, 0 = flat) in its neighborhood.
// Vertices are never moved, so every snapped point stays a vertex of the mesh; only the
// connectivity changes, which flattens the cusp/sliver artifacts a bad local triangulation
// produces. Each pass flips a maximal set of edges that share no face (so the flips are
// independent); it repeats until a pass makes no improving flip or max_passes is reached.
//
// threshold (radians) gates which neighborhoods are touched: a flip is made only where the worst
// bend in its quad already exceeds threshold, so a region smoother than that is left exactly as
// generated. threshold = 0 flips wherever the worst bend can be lowered at all.
inline Mesh smooth_by_flips(const Mesh& mesh, const Mat3& aniso = Mat3::Identity(),
                            double threshold = 0.0, int max_passes = 20) {
  using geometry::Point2;
  using geometry::Point3;
  using geometry::Vector2;
  using geometry::Vector3;
  constexpr auto kPi = 3.141592653589793;

  // Measure all geometry (normals, dihedrals, intersections) in the lattice's isotropic frame so
  // the flip choice respects an anisotropic resolution; only the output keeps world positions.
  geometry::Points3 v = geometry::transform_points<3>(aniso, mesh.vertices());
  Faces f = mesh.faces();

  auto normal = [&](const Face& t) {
    Vector3 e1 = v.row(t[1]) - v.row(t[0]);
    Vector3 e2 = v.row(t[2]) - v.row(t[0]);
    return Vector3(e1.cross(e2));
  };
  // The bend (0 = flat) between two faces; degenerate or back-to-back pairs read as the worst.
  auto bend = [&](const Face& a, const Face& b) {
    Vector3 na = normal(a);
    Vector3 nb = normal(b);
    auto da = na.norm();
    auto db = nb.norm();
    if (!(da > 0.0) || !(db > 0.0)) {
      return kPi;
    }
    return std::acos(std::max(-1.0, std::min(1.0, na.dot(nb) / (da * db))));
  };
  auto key = [](Index a, Index b) {
    auto lo = static_cast<std::uint64_t>(a < b ? a : b);
    auto hi = static_cast<std::uint64_t>(a < b ? b : a);
    return (lo << 32) | hi;
  };
  auto third = [](const Face& t, Index a, Index b) -> Index {
    for (auto x : t) {
      if (x != a && x != b) {
        return x;
      }
    }
    return -1;
  };
  auto shared = [](const Face& a, const Face& b) {
    auto n = 0;
    for (auto u : a) {
      for (auto w : b) {
        if (u == w) {
          n++;
        }
      }
    }
    return n;
  };
  // A real crossing of a and b beyond a shared vertex or edge. Two cases, split by how parallel
  // the faces are, because each case has a configuration the other mishandles:
  //  - Near-parallel: the 3D segment test reports a grazing line of contact between two nearly
  //    coplanar faces that merely tile (e.g. a fan around a shared vertex) as a crossing. Instead
  //    require b to straddle a's plane and their footprints to overlap in it (separating axis,
  //    exact for triangles), so a flat tiling is correctly not a crossing.
  //  - Transversal: project-and-separate is invalid, so use the 3D segment test; a point touch at
  //    a shared vertex yields a near-zero segment and stays below tol.
  auto overlaps = [&](const Face& a, const Face& b, double tol) {
    if (shared(a, b) >= 2) {
      return false;
    }
    Point3 a0 = v.row(a[0]);
    Point3 a1 = v.row(a[1]);
    Point3 a2 = v.row(a[2]);
    Point3 b0 = v.row(b[0]);
    Point3 b1 = v.row(b[1]);
    Point3 b2 = v.row(b[2]);
    Vector3 na = (a1 - a0).cross(a2 - a0);
    Vector3 nb = (b1 - b0).cross(b2 - b0);
    auto la = na.norm();
    auto lb = nb.norm();
    if (!(la > 0.0) || !(lb > 0.0)) {
      return false;
    }
    if (std::abs(na.dot(nb)) >= 0.9 * la * lb) {
      Vector3 n = na / la;
      auto dmin = std::numeric_limits<double>::infinity();
      auto dmax = -dmin;
      for (const Point3& p : {b0, b1, b2}) {
        auto d = n.dot(p - a0);
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
      }
      if (dmin > tol || dmax < -tol) {
        return false;  // b lies entirely on one side of a's plane
      }
      Vector3 u = (a1 - a0).normalized();
      Vector3 w = n.cross(u);
      auto proj = [&](const Point3& p) {
        Vector3 d = p - a0;
        return Point2(d.dot(u), d.dot(w));
      };
      std::array<Point2, 3> pa{proj(a0), proj(a1), proj(a2)};
      std::array<Point2, 3> pb{proj(b0), proj(b1), proj(b2)};
      // The axis is normalized so slack is a real distance: two faces sharing a vertex touch there
      // exactly, and that boundary contact must read as separated (not an overlap) despite the
      // round-off in the projection.
      auto separates = [](const std::array<Point2, 3>& p, const std::array<Point2, 3>& q,
                          double slack) {
        for (auto e = 0; e < 3; e++) {
          Vector2 edge = p.at((e + 1) % 3) - p.at(e);
          Vector2 axis(-edge.y(), edge.x());
          auto len = axis.norm();
          if (!(len > 0.0)) {
            continue;
          }
          axis /= len;
          auto pmin = std::numeric_limits<double>::infinity();
          auto pmax = -pmin;
          auto qmin = pmin;
          auto qmax = pmax;
          for (auto k = 0; k < 3; k++) {
            auto pp = axis.dot(p.at(k));
            auto qq = axis.dot(q.at(k));
            pmin = std::min(pmin, pp);
            pmax = std::max(pmax, pp);
            qmin = std::min(qmin, qq);
            qmax = std::max(qmax, qq);
          }
          if (pmax <= qmin + slack || qmax <= pmin + slack) {
            return true;
          }
        }
        return false;
      };
      return !(separates(pa, pb, tol) || separates(pb, pa, tol));
    }
    // Transversal: the 3D segment test, on triangles shrunk slightly toward their centroids so a
    // bare touch at a shared vertex (a legitimate crease) separates and is not reported as a
    // crossing, while a real overlap -- which has positive area -- survives the shrink.
    auto shrunk = [&](const Face& t) {
      Point3 p0 = v.row(t[0]);
      Point3 p1 = v.row(t[1]);
      Point3 p2 = v.row(t[2]);
      Point3 g = p0 + ((p1 - p0) + (p2 - p0)) / 3.0;
      constexpr auto s = 1e-6;
      return std::array<Point3, 3>{p0 + s * (g - p0), p1 + s * (g - p1), p2 + s * (g - p2)};
    };
    auto sa = shrunk(a);
    auto sb = shrunk(b);
    bool coplanar = false;
    Point3 s;
    Point3 t;
    if (!igl::tri_tri_intersection_test_3d(sa[0], sa[1], sa[2], sb[0], sb[1], sb[2], coplanar, s,
                                           t)) {
      return false;
    }
    return !coplanar && (t - s).norm() > tol;
  };

  for (auto pass = 0; pass < max_passes; pass++) {
    std::unordered_map<std::uint64_t, std::array<Index, 2>> ef;
    std::unordered_map<std::uint64_t, int> count;
    std::unordered_map<Index, std::vector<Index>> v2f;
    for (Index fi = 0; fi < f.rows(); fi++) {
      for (auto k = 0; k < 3; k++) {
        auto e = key(f(fi, k), f(fi, (k + 1) % 3));
        auto n = count[e]++;
        if (n < 2) {
          ef[e][n] = fi;
        }
        v2f[f(fi, k)].push_back(fi);
      }
    }
    auto external = [&](std::uint64_t e, Index f0, Index f1) -> Index {
      if (count[e] != 2) {
        return -1;
      }
      const auto& pr = ef[e];
      return pr[0] == f0 || pr[0] == f1 ? pr[1] : pr[0];
    };

    std::vector<bool> flipped(f.rows(), false);
    bool changed = false;
    for (const auto& [e, pr] : ef) {
      if (count[e] != 2) {
        continue;
      }
      Index f0i = pr[0];
      Index f1i = pr[1];
      if (flipped[f0i] || flipped[f1i]) {
        continue;
      }
      Face f0 = f.row(f0i);
      Face f1 = f.row(f1i);

      Index x = -1;
      Index y = -1;
      for (auto k = 0; k < 3; k++) {
        if (key(f0[k], f0[(k + 1) % 3]) == e) {
          x = f0[k];
          y = f0[(k + 1) % 3];
          break;
        }
      }
      Index c = third(f0, x, y);
      Index d = third(f1, x, y);
      if (c < 0 || d < 0 || c == d || count.contains(key(c, d))) {
        continue;  // boundary/degenerate, or the flipped diagonal already exists
      }

      Face nf0{x, d, c};
      Face nf1{d, y, c};
      Vector3 nn0 = normal(nf0);
      Vector3 nn1 = normal(nf1);
      if (!(nn0.norm() > 0.0) || !(nn1.norm() > 0.0)) {
        continue;
      }
      // Reject a flip that folds a triangle: its normal must not oppose the quad's (the sum of
      // the two old face normals). This keeps the surface from turning back on itself.
      Vector3 n0 = normal(f0);
      Vector3 n1 = normal(f1);
      auto d0 = n0.norm();
      auto d1 = n1.norm();
      if (d0 > 0.0 && d1 > 0.0) {
        Vector3 avg = n0 / d0 + n1 / d1;
        if (nn0.dot(avg) <= 0.0 || nn1.dot(avg) <= 0.0) {
          continue;
        }
      }

      Index g_cx = external(key(c, x), f0i, f1i);
      Index g_yc = external(key(y, c), f0i, f1i);
      Index g_xd = external(key(x, d), f0i, f1i);
      Index g_dy = external(key(d, y), f0i, f1i);
      auto wb = [&](const Face& a, Index gi) { return gi < 0 ? 0.0 : bend(a, f.row(gi)); };

      auto before =
          std::max({bend(f0, f1), wb(f0, g_cx), wb(f0, g_yc), wb(f1, g_xd), wb(f1, g_dy)});
      auto after =
          std::max({bend(nf0, nf1), wb(nf0, g_cx), wb(nf1, g_yc), wb(nf0, g_xd), wb(nf1, g_dy)});
      if (before > threshold && after < before - 1e-6) {
        // Local self-intersection guard: neither new triangle may cross a face in the quad's
        // vertex one-ring (catches a flipped diagonal that passes over a nearby sheet).
        auto tol = 1e-6 * (v.row(x) - v.row(y)).norm();
        bool safe = true;
        for (Index ring : {x, y, c, d}) {
          for (Index gi : v2f[ring]) {
            if (gi != f0i && gi != f1i &&
                (overlaps(nf0, f.row(gi), tol) || overlaps(nf1, f.row(gi), tol))) {
              safe = false;
              break;
            }
          }
          if (!safe) {
            break;
          }
        }
        if (!safe) {
          continue;
        }
        f.row(f0i) = nf0;
        f.row(f1i) = nf1;
        flipped[f0i] = true;
        flipped[f1i] = true;
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }

  return {mesh.vertices(), std::move(f)};  // world positions, only connectivity changed
}

}  // namespace polatory::isosurface
