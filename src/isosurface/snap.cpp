#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <polatory/isosurface/smooth.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/isosurface/snapper/self_intersection.hpp>
#include <polatory/isosurface/snapper/snapper.hpp>
#include <unordered_set>

namespace polatory::isosurface {

namespace {

// Quantize a position to an integer key so a snapped vertex (which sits exactly on its snap
// point, up to the aniso round-trip) can be matched back to the point's index.
std::array<long long, 3> pos_key(double x, double y, double z) {
  constexpr double q = 1e-9;
  return {std::llround(x / q), std::llround(y / q), std::llround(z / q)};
}

// EXPERIMENT (POLATORY_RELAX): snap with the self-intersection proxies relaxed, then a real
// global self-intersection test culls the points that actually cross. Runs a fixed snap_iter
// number of iterations: iteration k activates the points with priority <= k (so priority 0 is
// snapped first and lower-priority points are introduced progressively), minus the points culled
// by the self-intersection test in earlier iterations.
Mesh snap_relaxed(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                  const geometry::Bbox3& bbox, double max_distance, const Mat3& aniso,
                  const VecX& priorities, int snap_iter, bool smooth) {
  std::map<std::array<long long, 3>, Index> point_index;
  for (Index i = 0; i < points.rows(); i++) {
    point_index[pos_key(points(i, 0), points(i, 1), points(i, 2))] = i;
  }
  bool has_priority = priorities.size() == points.rows();

  std::unordered_set<Index> culled;  // points that self-intersected; carried across iterations
  Mesh out;
  for (int it = 0; it < std::max(1, snap_iter); it++) {
    std::unordered_set<Index> exclude = culled;
    if (has_priority) {
      for (Index i = 0; i < points.rows(); i++) {
        if (priorities(i) > static_cast<double>(it)) {
          exclude.insert(i);  // not yet introduced at this priority level
        }
      }
    }
    auto active = points.rows() - static_cast<Index>(exclude.size());

    snapper::Snapper s(mesh.vertices(), mesh.faces(), bbox, max_distance, aniso);
    out = s.snap(points, tolerances, &exclude);
    const auto& relaxed = s.relaxed_captured();

    // Smooth before the self-intersection test so flips can resolve some crossings; vertices
    // are unmoved, so the position-based point matching below still holds.
    if (smooth) {
      out = smooth_by_flips(out, aniso);
    }

    auto bad = snapper::self_intersecting_faces(out.vertices(), out.faces());
    auto before = culled.size();
    auto cull = [&](bool relaxed_only) {
      for (auto f : bad) {
        for (auto k = 0; k < 3; k++) {
          Index v = out.faces()(f, k);
          auto pi = point_index.find(
              pos_key(out.vertices()(v, 0), out.vertices()(v, 1), out.vertices()(v, 2)));
          if (pi != point_index.end() && (!relaxed_only || relaxed.contains(pi->second))) {
            culled.insert(pi->second);
          }
        }
      }
    };
    cull(/*relaxed_only=*/true);
    if (culled.size() == before && !bad.empty()) {
      cull(/*relaxed_only=*/false);
    }
    std::fprintf(stderr,
                 "[relax] iter=%d active=%lld dropped=%lld self_int_faces=%zu culled=%zu\n", it,
                 static_cast<long long>(active), static_cast<long long>(s.stats().dropped),
                 bad.size(), culled.size());
  }
  return out;
}

}  // namespace

Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               const geometry::Bbox3& bbox, double max_distance, const Mat3& aniso,
               const VecX& priorities, int snap_iter) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  // EXPERIMENT (POLATORY_SMOOTH): flatten cusp/sliver artifacts by edge flips that lower the
  // worst dihedral, without moving any vertex (so all snapped points stay honored).
  bool smooth = std::getenv("POLATORY_SMOOTH") != nullptr;

  if (std::getenv("POLATORY_RELAX") != nullptr) {
    // The relaxed path smooths per iteration (before each self-intersection cull).
    return snap_relaxed(mesh, points, tolerances, bbox, max_distance, aniso, priorities, snap_iter,
                        smooth);
  }

  snapper::Snapper s(mesh.vertices(), mesh.faces(), bbox, max_distance, aniso);
  auto out = s.snap(points, tolerances);
  if (smooth) {
    out = smooth_by_flips(out, aniso);
  }
  return out;
}

}  // namespace polatory::isosurface
