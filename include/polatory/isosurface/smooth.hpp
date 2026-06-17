#pragma once

#include <igl/tri_tri_intersect.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <cstdint>
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
  // A real crossing of a and b beyond a shared vertex or edge.
  auto overlaps = [&](const Face& a, const Face& b, double tol) {
    if (shared(a, b) >= 2) {
      return false;
    }
    bool coplanar = false;
    Eigen::RowVector3d s;
    Eigen::RowVector3d t;
    if (!igl::tri_tri_intersection_test_3d(v.at(a[0]), v.at(a[1]), v.at(a[2]), v.at(b[0]),
                                           v.at(b[1]), v.at(b[2]), coplanar, s, t)) {
      return false;
    }
    return coplanar || (t - s).norm() > tol;
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
