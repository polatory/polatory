#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
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
  static constexpr double kPi = 3.141592653589793;

  // A cell of the uniform spatial grid the self-intersection guard broad-phases over.
  using Cell = std::array<int, 3>;
  struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept {
      auto h = static_cast<std::size_t>(static_cast<std::uint32_t>(c[0])) * 73856093U;
      h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(c[1])) * 19349663U;
      h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(c[2])) * 83492791U;
      return h;
    }
  };

 public:
  // snapped, if non-empty, has one entry per vertex; a flip is then made only where its quad
  // touches a snapped vertex, leaving the rest of the mesh exactly as generated.
  Smoother(const Mesh& mesh, const Mat3& aniso, double threshold, int max_passes,
           std::vector<bool> snapped = {})
      : v_(geometry::transform_points<3>(aniso, mesh.vertices())),
        f_(mesh.faces()),
        threshold_(threshold),
        max_passes_(max_passes),
        snapped_(std::move(snapped)) {
    // The grid cell is the longest edge, so a face spans about one cell and its AABB touches few.
    for (Index fi = 0; fi < f_.rows(); fi++) {
      for (auto k = 0; k < 3; k++) {
        cell_ = std::max(cell_, (v_.row(f_(fi, k)) - v_.row(f_(fi, (k + 1) % 3))).norm());
      }
    }
    if (!(cell_ > 0.0)) {
      cell_ = 1.0;
    }

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
        Edge e{f_(fi, k), f_(fi, (k + 1) % 3)};
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

    // Index every face by the grid cells its AABB touches, so the guard can broad-phase against
    // spatially near faces rather than only the quad's one-ring. Vertices never move, so a face's
    // cells change only when it is flipped, when it is re-inserted in place (old entries left
    // stale are harmless: a stale index still reads its current geometry, and a dropped one is
    // re-found through the flipped face that now covers its region).
    auto cell_of = [&](const Point3& p) -> Cell {
      return {static_cast<int>(std::floor(p.x() / cell_)),
              static_cast<int>(std::floor(p.y() / cell_)),
              static_cast<int>(std::floor(p.z() / cell_))};
    };
    std::unordered_map<Cell, std::vector<Index>, CellHash> grid;
    auto insert_face = [&](Index fi) {
      Point3 p0 = v_.row(f_(fi, 0));
      Point3 p1 = v_.row(f_(fi, 1));
      Point3 p2 = v_.row(f_(fi, 2));
      Cell lo = cell_of(p0.cwiseMin(p1).cwiseMin(p2));
      Cell hi = cell_of(p0.cwiseMax(p1).cwiseMax(p2));
      for (auto i = lo[0]; i <= hi[0]; i++) {
        for (auto j = lo[1]; j <= hi[1]; j++) {
          for (auto k = lo[2]; k <= hi[2]; k++) {
            grid[Cell{i, j, k}].push_back(fi);
          }
        }
      }
    };
    for (Index fi = 0; fi < f_.rows(); fi++) {
      insert_face(fi);
    }
    std::vector<Index> visited(f_.rows(), -1);  // per-guard stamp, to test each candidate once
    Index guard_id = 0;

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
        if (e == Edge{f0[k], f0[(k + 1) % 3]}) {
          x = f0[k];
          y = f0[(k + 1) % 3];
          break;
        }
      }
      Index c = third(f0, e);
      Index d = third(f1, e);
      if (c < 0 || d < 0 || c == d || count.contains({c, d})) {
        continue;  // boundary/degenerate, or the flipped diagonal already exists
      }
      if (!snapped_.empty() &&
          !(snapped_[x] || snapped_[y] || snapped_[c] || snapped_[d])) {
        continue;  // restricted to quads touching a snapped vertex
      }

      Face nf0{x, d, c};
      Face nf1{d, y, c};
      Vector3 nn0 = normal(nf0);
      Vector3 nn1 = normal(nf1);
      if (!(nn0.norm() > 0.0) || !(nn1.norm() > 0.0)) {
        continue;
      }

      // Reject a flip that folds the surface back on itself: a new normal must not oppose the
      // quad normal (the sum of the two old face normals).
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

      Index g_cx = external({c, x}, f0i, f1i);
      Index g_yc = external({y, c}, f0i, f1i);
      Index g_xd = external({x, d}, f0i, f1i);
      Index g_dy = external({d, y}, f0i, f1i);

      auto before = std::max({bend(f0, f1), bend_with(f0, g_cx), bend_with(f0, g_yc),
                              bend_with(f1, g_xd), bend_with(f1, g_dy)});
      auto after = std::max({bend(nf0, nf1), bend_with(nf0, g_cx), bend_with(nf1, g_yc),
                             bend_with(nf0, g_xd), bend_with(nf1, g_dy)});
      if (before > threshold_ && after < before - 1e-6) {
        // Self-intersection guard: neither new triangle may cross any face whose grid cell its
        // AABB touches -- a spatial broad-phase, so a flipped diagonal that passes over a
        // topologically distant but spatially near sheet is caught, not just one in the one-ring.
        auto tol = 1e-6 * (v_.row(x) - v_.row(y)).norm();
        guard_id++;
        auto crosses = [&](const Face& nf) {
          Point3 p0 = v_.row(nf[0]);
          Point3 p1 = v_.row(nf[1]);
          Point3 p2 = v_.row(nf[2]);
          Cell lo = cell_of(p0.cwiseMin(p1).cwiseMin(p2));
          Cell hi = cell_of(p0.cwiseMax(p1).cwiseMax(p2));
          for (auto i = lo[0]; i <= hi[0]; i++) {
            for (auto j = lo[1]; j <= hi[1]; j++) {
              for (auto k = lo[2]; k <= hi[2]; k++) {
                auto it = grid.find(Cell{i, j, k});
                if (it == grid.end()) {
                  continue;
                }
                for (Index gi : it->second) {
                  if (visited[gi] == guard_id) {
                    continue;
                  }
                  visited[gi] = guard_id;
                  if (gi != f0i && gi != f1i &&
                      (overlaps(nf0, f_.row(gi), tol) || overlaps(nf1, f_.row(gi), tol))) {
                    return true;
                  }
                }
              }
            }
          }
          return false;
        };
        if (crosses(nf0) || crosses(nf1)) {
          continue;
        }
        f_.row(f0i) = nf0;
        f_.row(f1i) = nf1;
        insert_face(f0i);
        insert_face(f1i);
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

  // A face's unnormalized normal (length is twice the area).
  Vector3 normal(const Face& t) const {
    Vector3 e1 = v_.row(t[1]) - v_.row(t[0]);
    Vector3 e2 = v_.row(t[2]) - v_.row(t[0]);
    return Vector3(e1.cross(e2));
  }

  // A real crossing of a and b beyond a shared vertex (see triangles_overlap_3d). Edge-adjacent
  // pairs (shared >= 2) are skipped: a flip cannot fold the surface back on an edge without either
  // opposing the quad normal (rejected above) or worsening a quad bend (so it is not an
  // improvement and is not made), so the only crossings left to catch are with non-adjacent faces.
  bool overlaps(const Face& a, const Face& b, double tol) const {
    if (shared_vertices(a, b) >= 2) {
      return false;
    }
    Point3 a0 = v_.row(a[0]);
    Point3 a1 = v_.row(a[1]);
    Point3 a2 = v_.row(a[2]);
    Point3 b0 = v_.row(b[0]);
    Point3 b1 = v_.row(b[1]);
    Point3 b2 = v_.row(b[2]);
    return triangles_overlap_3d(a0, a1, a2, b0, b1, b2, tol);
  }

  static int shared_vertices(const Face& a, const Face& b) {
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

  // The vertex of f that is neither endpoint of e, or -1 if there is none.
  static Index third(const Face& f, const Edge& e) {
    for (auto x : f) {
      if (x != e.a && x != e.b) {
        return x;
      }
    }
    return -1;
  }

  geometry::Points3 v_;  // vertices in the isotropic frame, where the geometry is measured
  Faces f_;              // working faces; connectivity is edited in place
  double cell_{};        // spatial-grid cell size for the self-intersection guard
  double threshold_;
  int max_passes_;
  std::vector<bool> snapped_;  // if non-empty, restrict flips to quads touching a snapped vertex
  Mesh mesh_;
};

// The post-snap cleanup: flatten the snapped region by edge flips (see Smoother), so that the
// cusp/sliver artifacts the snapped triangulation leaves are smoothed wherever a crease exceeds
// 5 degrees, while the rest of the mesh is left exactly as generated. aniso maps world into the
// isotropic frame. snapped, if non-empty (one entry per vertex), restricts flips to quads touching
// a snapped vertex; when empty, nothing is flipped. Vertices never move, so every snap point stays.
inline Mesh post_snap(const Mesh& mesh, const Mat3& aniso = Mat3::Identity(),
                      std::vector<bool> snapped = {}, int max_passes = 20) {
  constexpr double kDegree = 0.017453292519943295;  // radians
  return Smoother(mesh, aniso, 5.0 * kDegree, max_passes, std::move(snapped)).mesh();
}

}  // namespace polatory::isosurface
