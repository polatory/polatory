#pragma once

#include <igl/tri_tri_intersect.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <boost/container_hash/hash.hpp>
#include <cmath>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/predicates.hpp>
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
// All geometry is measured in the lattice's isotropic frame (aniso maps world into it) so the flip
// choice respects an anisotropic resolution; only the output keeps world positions. threshold
// (radians) gates which neighborhoods are touched: a flip is made only where the worst bend in its
// quad already exceeds threshold, so a region smoother than that is left exactly as generated.
// threshold = 0 flips wherever the worst bend can be lowered at all.
class Smoother {
  using Point2 = geometry::Point2;
  using Point3 = geometry::Point3;
  using Vector3 = geometry::Vector3;
  using Edge = std::pair<Index, Index>;
  using EdgeHash = boost::hash<Edge>;

  static constexpr double kPi = 3.141592653589793;

 public:
  Smoother(const Mesh& mesh, const Mat3& aniso, double threshold, int max_passes)
      : v_(geometry::transform_points<3>(aniso, mesh.vertices())),
        f_(mesh.faces()),
        threshold_(threshold),
        max_passes_(max_passes) {
    std::vector<bool> dirty(f_.rows(), true);  // every face is worth examining in the first pass
    for (auto pass = 0; pass < max_passes_; pass++) {
      if (!flip_pass(dirty)) {
        break;
      }
    }
    mesh_ = {mesh.vertices(), std::move(f_)};
  }

  Mesh mesh() && { return std::move(mesh_); }

 private:
  // The bend (0 = flat) between two faces; degenerate or back-to-back pairs read as the worst.
  double bend(const Face& a, const Face& b) const {
    Vector3 na = normal(a);
    Vector3 nb = normal(b);
    auto da = na.norm();
    auto db = nb.norm();
    if (!(da > 0.0) || !(db > 0.0)) {
      return kPi;
    }
    return std::acos(std::clamp(na.dot(nb) / (da * db), -1.0, 1.0));
  }

  // bend(a, face fi), or 0 when fi is the absent neighbour across a boundary edge (fi < 0).
  double bend_with(const Face& a, Index fi) const { return fi < 0 ? 0.0 : bend(a, f_.row(fi)); }

  // One pass: flip a maximal set of improving edges that share no face. Only edges with a face
  // dirty since the last pass (changed, or beside a change) are re-examined -- the rest cannot have
  // become flippable -- and dirty is then replaced with the faces this pass touched. Returns
  // whether any edge was flipped (the constructor loops until a pass flips nothing).
  bool flip_pass(std::vector<bool>& dirty) {
    std::unordered_map<Edge, std::array<Index, 2>, EdgeHash> ef;
    std::unordered_map<Edge, int, EdgeHash> count;
    std::unordered_map<Index, std::vector<Index>> v2f;
    for (Index fi = 0; fi < f_.rows(); fi++) {
      for (auto k = 0; k < 3; k++) {
        auto e = make_edge(f_(fi, k), f_(fi, (k + 1) % 3));
        auto n = count[e]++;
        if (n < 2) {
          ef[e][n] = fi;
        }
        v2f[f_(fi, k)].push_back(fi);
      }
    }
    auto external = [&](const Edge& e, Index f0, Index f1) -> Index {
      if (count[e] != 2) {
        return -1;
      }
      const auto& pr = ef[e];
      return pr[0] == f0 || pr[0] == f1 ? pr[1] : pr[0];
    };

    std::vector<bool> flipped(f_.rows(), false);
    std::vector<bool> next_dirty(f_.rows(), false);
    bool changed = false;
    for (const auto& [e, pr] : ef) {
      if (count[e] != 2) {
        continue;
      }
      Index f0i = pr[0];
      Index f1i = pr[1];
      if ((!dirty[f0i] && !dirty[f1i]) || flipped[f0i] || flipped[f1i]) {
        continue;
      }
      Face f0 = f_.row(f0i);
      Face f1 = f_.row(f1i);

      Index x = -1;
      Index y = -1;
      for (auto k = 0; k < 3; k++) {
        if (make_edge(f0[k], f0[(k + 1) % 3]) == e) {
          x = f0[k];
          y = f0[(k + 1) % 3];
          break;
        }
      }
      Index c = third(f0, x, y);
      Index d = third(f1, x, y);
      if (c < 0 || d < 0 || c == d || count.contains(make_edge(c, d))) {
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

      Index g_cx = external(make_edge(c, x), f0i, f1i);
      Index g_yc = external(make_edge(y, c), f0i, f1i);
      Index g_xd = external(make_edge(x, d), f0i, f1i);
      Index g_dy = external(make_edge(d, y), f0i, f1i);

      auto before = std::max({bend(f0, f1), bend_with(f0, g_cx), bend_with(f0, g_yc),
                              bend_with(f1, g_xd), bend_with(f1, g_dy)});
      auto after = std::max({bend(nf0, nf1), bend_with(nf0, g_cx), bend_with(nf1, g_yc),
                             bend_with(nf0, g_xd), bend_with(nf1, g_dy)});
      if (before > threshold_ && after < before - 1e-6) {
        // Local self-intersection guard: neither new triangle may cross a face in the quad's
        // vertex one-ring (catches a flipped diagonal that passes over a nearby sheet).
        auto tol = 1e-6 * (v_.row(x) - v_.row(y)).norm();
        bool safe = true;
        for (Index ring : {x, y, c, d}) {
          for (Index gi : v2f[ring]) {
            if (gi != f0i && gi != f1i &&
                (overlaps(nf0, f_.row(gi), tol) || overlaps(nf1, f_.row(gi), tol))) {
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
        f_.row(f0i) = nf0;
        f_.row(f1i) = nf1;
        flipped[f0i] = true;
        flipped[f1i] = true;
        changed = true;
        // The flip rewired the quad's vertices, so every face around them is worth re-examining.
        for (Index ring : {x, y, c, d}) {
          for (Index g : v2f[ring]) {
            next_dirty[g] = true;
          }
        }
      }
    }
    dirty = std::move(next_dirty);
    return changed;
  }

  static Edge make_edge(Index a, Index b) { return a < b ? Edge{a, b} : Edge{b, a}; }

  // The doubled-area normal of a face (its length is twice the area).
  Vector3 normal(const Face& t) const {
    Vector3 e1 = v_.row(t[1]) - v_.row(t[0]);
    Vector3 e2 = v_.row(t[2]) - v_.row(t[0]);
    return Vector3(e1.cross(e2));
  }

  // A real crossing of a and b beyond a shared vertex or edge. Two cases, split by how parallel
  // the faces are, because each case has a configuration the other mishandles:
  //  - Near-parallel: the 3D segment test reports a grazing line of contact between two nearly
  //    coplanar faces that merely tile (e.g. a fan around a shared vertex) as a crossing. Instead
  //    require b to straddle a's plane and their footprints to overlap in it (separating axis,
  //    exact for triangles), so a flat tiling is correctly not a crossing.
  //  - Transversal: project-and-separate is invalid, so use the 3D segment test; a point touch at
  //    a shared vertex yields a near-zero segment and stays below tol.
  bool overlaps(const Face& a, const Face& b, double tol) const {
    if (shared(a, b) >= 2) {
      return false;
    }
    Point3 a0 = v_.row(a[0]);
    Point3 a1 = v_.row(a[1]);
    Point3 a2 = v_.row(a[2]);
    Point3 b0 = v_.row(b[0]);
    Point3 b1 = v_.row(b[1]);
    Point3 b2 = v_.row(b[2]);
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
      // The footprints overlap in a's plane (slack = tol, a real distance, so a shared-vertex
      // touch reads as separated rather than an overlap).
      return triangles_overlap_2d(pa, pb, tol);
    }
    // Transversal: the 3D segment test, on triangles shrunk slightly toward their centroids so a
    // bare touch at a shared vertex (a legitimate crease) separates and is not reported as a
    // crossing, while a real overlap -- which has positive area -- survives the shrink.
    auto shrunk = [&](const Face& t) {
      Point3 p0 = v_.row(t[0]);
      Point3 p1 = v_.row(t[1]);
      Point3 p2 = v_.row(t[2]);
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
  }

  // The number of vertices a and b share.
  static int shared(const Face& a, const Face& b) {
    auto n = 0;
    for (auto u : a) {
      for (auto w : b) {
        if (u == w) {
          n++;
        }
      }
    }
    return n;
  }

  // The vertex of f that is neither a nor b, or -1 if there is none.
  static Index third(const Face& f, Index a, Index b) {
    for (auto x : f) {
      if (x != a && x != b) {
        return x;
      }
    }
    return -1;
  }

  geometry::Points3 v_;  // vertices in the isotropic frame, where the geometry is measured
  Faces f_;              // working faces; connectivity is edited in place
  double threshold_;
  int max_passes_;
  Mesh mesh_;  // the smoothed result
};

// Smooths the mesh by edge flips (see Smoother). aniso maps world into the isotropic frame;
// threshold (radians) is the crease angle above which a neighborhood is flattened.
inline Mesh smooth_by_flips(const Mesh& mesh, const Mat3& aniso = Mat3::Identity(),
                            double threshold = 0.0, int max_passes = 20) {
  return Smoother(mesh, aniso, threshold, max_passes).mesh();
}

}  // namespace polatory::isosurface
