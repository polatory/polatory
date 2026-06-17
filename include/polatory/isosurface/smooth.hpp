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
// worst dihedral (the largest bend between adjacent faces) in its neighborhood. Vertices are
// never moved, so every snapped point stays a vertex of the mesh; only the connectivity changes,
// which flattens the cusp/sliver artifacts a bad local triangulation produces. Each pass flips a
// maximal set of edges that share no face (so the flips are independent); it repeats until a pass
// makes no improving flip or max_passes is reached.
inline Mesh smooth_by_flips(const Mesh& mesh, const Mat3& aniso = Mat3::Identity(),
                            int max_passes = 20) {
  using geometry::Point3;
  using geometry::Vector3;
  using Face = std::array<Index, 3>;
  constexpr double kPi = 3.14159265358979323846;

  // Measure all geometry (normals, dihedrals, intersections) in the lattice's isotropic frame so
  // the flip choice respects an anisotropic resolution; only the output keeps world positions.
  geometry::Points3 iso = mesh.vertices() * aniso.transpose();
  std::vector<Point3> v;
  v.reserve(iso.rows());
  for (auto r : iso.rowwise()) {
    v.emplace_back(r);
  }
  std::vector<Face> f;
  f.reserve(mesh.faces().rows());
  for (auto r : mesh.faces().rowwise()) {
    f.push_back({r(0), r(1), r(2)});
  }

  auto normal = [&](const Face& t) {
    Vector3 e1 = v.at(t[1]) - v.at(t[0]);
    Vector3 e2 = v.at(t[2]) - v.at(t[0]);
    return Vector3(e1.cross(e2));
  };
  // The bend (0 = flat) between two faces; degenerate or back-to-back pairs read as the worst.
  auto bend = [&](const Face& a, const Face& b) {
    Vector3 na = normal(a);
    Vector3 nb = normal(b);
    double da = na.norm();
    double db = nb.norm();
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
    int n = 0;
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
    Vector3 a0 = v.at(a[0]);
    Vector3 a1 = v.at(a[1]);
    Vector3 a2 = v.at(a[2]);
    Vector3 b0 = v.at(b[0]);
    Vector3 b1 = v.at(b[1]);
    Vector3 b2 = v.at(b[2]);
    Vector3 na = (a1 - a0).cross(a2 - a0);
    Vector3 nb = (b1 - b0).cross(b2 - b0);
    double la = na.norm();
    double lb = nb.norm();
    if (!(la > 0.0) || !(lb > 0.0)) {
      return false;
    }
    if (std::abs(na.dot(nb)) >= 0.9 * la * lb) {
      Vector3 n = na / la;
      double dmin = std::numeric_limits<double>::infinity();
      double dmax = -dmin;
      for (const Vector3& p : {b0, b1, b2}) {
        double d = n.dot(p - a0);
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
      }
      if (dmin > tol || dmax < -tol) {
        return false;  // b lies entirely on one side of a's plane
      }
      Vector3 u = (a1 - a0).normalized();
      Vector3 w = n.cross(u);
      auto proj = [&](const Vector3& p) {
        Vector3 d = p - a0;
        return std::array<double, 2>{d.dot(u), d.dot(w)};
      };
      std::array<std::array<double, 2>, 3> pa{proj(a0), proj(a1), proj(a2)};
      std::array<std::array<double, 2>, 3> pb{proj(b0), proj(b1), proj(b2)};
      // The axis is normalized so slack is a real distance: two faces sharing a vertex touch there
      // exactly, and that boundary contact must read as separated (not an overlap) despite the
      // round-off in the projection.
      auto separates = [](const std::array<std::array<double, 2>, 3>& p,
                          const std::array<std::array<double, 2>, 3>& q, double slack) {
        for (int e = 0; e < 3; e++) {
          std::array<double, 2> axis{p.at(e).at(1) - p.at((e + 1) % 3).at(1),
                                     p.at((e + 1) % 3).at(0) - p.at(e).at(0)};
          double al = std::hypot(axis.at(0), axis.at(1));
          if (!(al > 0.0)) {
            continue;
          }
          axis.at(0) /= al;
          axis.at(1) /= al;
          double pmin = std::numeric_limits<double>::infinity();
          double pmax = -pmin;
          double qmin = pmin;
          double qmax = pmax;
          for (int k = 0; k < 3; k++) {
            double pp = axis.at(0) * p.at(k).at(0) + axis.at(1) * p.at(k).at(1);
            double qq = axis.at(0) * q.at(k).at(0) + axis.at(1) * q.at(k).at(1);
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
      Eigen::RowVector3d p0 = v.at(t[0]);
      Eigen::RowVector3d p1 = v.at(t[1]);
      Eigen::RowVector3d p2 = v.at(t[2]);
      Eigen::RowVector3d g = (p0 + p1 + p2) / 3.0;
      constexpr double s = 1e-6;
      return std::array<Eigen::RowVector3d, 3>{p0 + s * (g - p0), p1 + s * (g - p1),
                                               p2 + s * (g - p2)};
    };
    auto sa = shrunk(a);
    auto sb = shrunk(b);
    bool coplanar = false;
    Eigen::RowVector3d s;
    Eigen::RowVector3d t;
    if (!igl::tri_tri_intersection_test_3d(sa[0], sa[1], sa[2], sb[0], sb[1], sb[2], coplanar, s,
                                           t)) {
      return false;
    }
    return !coplanar && (t - s).norm() > tol;
  };

  for (int pass = 0; pass < max_passes; pass++) {
    std::unordered_map<std::uint64_t, std::array<Index, 2>> ef;
    std::unordered_map<std::uint64_t, int> count;
    std::unordered_map<Index, std::vector<Index>> v2f;
    for (Index fi = 0; fi < static_cast<Index>(f.size()); fi++) {
      for (int k = 0; k < 3; k++) {
        auto e = key(f.at(fi)[k], f.at(fi)[(k + 1) % 3]);
        int n = count[e]++;
        if (n < 2) {
          ef[e][n] = fi;
        }
        v2f[f.at(fi)[k]].push_back(fi);
      }
    }
    auto external = [&](std::uint64_t e, Index f0, Index f1) -> Index {
      if (count[e] != 2) {
        return -1;
      }
      const auto& pr = ef[e];
      return pr[0] == f0 || pr[0] == f1 ? pr[1] : pr[0];
    };

    std::vector<bool> flipped(f.size(), false);
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
      const Face& f0 = f.at(f0i);
      const Face& f1 = f.at(f1i);

      Index x = -1;
      Index y = -1;
      for (int k = 0; k < 3; k++) {
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
      double d0 = n0.norm();
      double d1 = n1.norm();
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
      auto wb = [&](const Face& a, Index gi) { return gi < 0 ? 0.0 : bend(a, f.at(gi)); };

      double before = std::max({bend(f0, f1), wb(f0, g_cx), wb(f0, g_yc), wb(f1, g_xd),
                                wb(f1, g_dy)});
      double after = std::max({bend(nf0, nf1), wb(nf0, g_cx), wb(nf1, g_yc), wb(nf0, g_xd),
                               wb(nf1, g_dy)});
      if (after < before - 1e-6) {
        // Local self-intersection guard: neither new triangle may cross a face in the quad's
        // vertex one-ring (catches a flipped diagonal that passes over a nearby sheet).
        double tol = 1e-6 * (v.at(x) - v.at(y)).norm();
        bool safe = true;
        for (Index ring : {x, y, c, d}) {
          for (Index gi : v2f[ring]) {
            if (gi != f0i && gi != f1i &&
                (overlaps(nf0, f.at(gi), tol) || overlaps(nf1, f.at(gi), tol))) {
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
        f.at(f0i) = nf0;
        f.at(f1i) = nf1;
        flipped[f0i] = true;
        flipped[f1i] = true;
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }

  Faces out_f(static_cast<Index>(f.size()), 3);
  for (Index i = 0; i < out_f.rows(); i++) {
    out_f(i, 0) = f.at(i)[0];
    out_f(i, 1) = f.at(i)[1];
    out_f(i, 2) = f.at(i)[2];
  }
  return {mesh.vertices(), std::move(out_f)};  // world positions, only connectivity changed
}

}  // namespace polatory::isosurface
