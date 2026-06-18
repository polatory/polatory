#pragma once

#include <igl/barycentric_coordinates.h>
#include <igl/tri_tri_intersect.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/predicates.hpp>
#include <polatory/isosurface/snapper/original_mesh.hpp>
#include <polatory/isosurface/snapper/triangulation.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace polatory::isosurface::snapper {

// Snaps a mesh to a subset of the given points so that the mesh passes through them without
// becoming self-intersecting. Each point is processed in turn and either snapped or dropped;
// the result provably has no self-intersections and interpolates as many points as it can.
//
// Each point goes through three steps:
//
//  - Classify. Match the point, once and against the original mesh, to the nearest simplex of
//    its closest face — a vertex, an edge, or the face interior — by which simplex centroid
//    (the vertex, an edge midpoint, or the face centroid) is nearest its projection (a Voronoi
//    classification of the face).
//
//  - Snap. A vertex match reuses the existing vertex (a move); an edge match adds a vertex on
//    the subdivided original edge shared by the two incident patches; a face match adds a
//    vertex interior to one patch. Each affected patch — an original face with vertices added
//    on its edges and interior — is then re-triangulated from scratch by a constrained
//    Delaunay triangulation (see Triangulation) over the faithful (off-surface) emitted
//    positions, with the subdivided original edges as boundary constraints.
//
//  - Accept or drop. Keep the snap only if it leaves the mesh free of self-intersections: the
//    affected patches are tested against their neighborhood and rejected if any new triangle
//    meets a non-adjacent one. A rejected vertex or edge snap cascades to its next-nearest
//    simplex (ultimately a face snap, which perturbs only its own patch); a point that cannot
//    be placed anywhere is dropped.
//
// Processing order. Points are taken by increasing tangential offset — the in-surface distance
// from a point's projection to the feature it snaps to. A vertex move's distortion is its
// tangential component (a perpendicular move only raises or lowers the height field, leaving
// the patch shape unchanged), so taking the least-tangential snaps first lets each shared
// vertex be claimed by its least-distorting candidate.
//
// Boundary. A point is snapped only if both it and its projection lie inside the given bbox.
// If the bbox interior holds no mesh boundary, snapping provably leaves the boundary untouched
// (no boundary vertex moved, no boundary edge subdivided), so boundary features need no
// special handling. See the comment on the constructor.
class Snapper {
  // A face as a vertex-index triple (the std::array form of the Eigen-row `Face`, used as
  // the element type of the patch triangulations).
  using Face = std::array<Index, 3>;

  // A simplex of a candidate's projected face that the point may be snapped to. The
  // enumerators are laid out vertices, then edges, then the face, and their underlying
  // values double as indices into the per-face site arrays: kVertex0..kVertex2 are the
  // vertices (0..2), kEdge12/kEdge20/kEdge01 the edge between the two named vertices (3..5),
  // kFace the interior (6).
  enum class Simplex { kVertex0, kVertex1, kVertex2, kEdge12, kEdge20, kEdge01, kFace };

  static int index_of(Simplex s) { return static_cast<int>(s); }

  // A point to be snapped, with the data needed to assign it to a simplex of its
  // projected face.
  struct Candidate {
    Point3 p;
    double d2{};                // The squared distance from p to the mesh.
    double tang{};              // The squared tangential offset: q to its snapped feature.
    double min_distance{};      // Skip if the mesh already passes within this of p (0 = never).
    Index face{};               // The projected face.
    std::array<Index, 3> fv{};  // Its vertex indices.
    std::array<double, 3> l{};  // The barycentric coordinates of the projection.
    std::array<Simplex, 7> order{};  // The seven simplices, nearest centroid first.
  };

  // A vertex on an edge, at parameter t from the edge's smaller-id endpoint.
  struct EdgeVertex {
    double t{};
    Index id{};
  };

 public:
  struct Stats {
    Index skipped{};    // outside bbox or beyond max_distance
    Index satisfied{};  // already within tolerance of the snapped mesh, so not attempted
    Index dropped{};    // classified but could not be placed without self-intersection
    Index moved_vertices{};
    Index inserted_on_edges{};
    Index inserted_in_faces{};
    Index pierce_rejections{};  // snaps rejected because they would pierce a distant face
    Index thinned_on_edges{};   // redundant inserted edge vertices dropped within tolerance
    Index thinned_in_faces{};   // redundant inserted interior vertices dropped within tolerance
  };

  // A point is snapped only if its distance to the mesh is at most max_distance and
  // both the point and its projection lie inside bbox.
  //
  // The vertices, points, and bbox are given in world space. The snapper measures all
  // distances — the projection onto the mesh, the max_distance bound, the simplex
  // classification, the processing order — in the lattice's isotropic frame, so that an
  // anisotropic resolution is respected rather than skewed by world space. aniso maps world
  // into that frame: the snapper transforms the vertices and points by it, snaps, and
  // transforms the emitted mesh back (aniso defaults to identity, a no-op). bbox stays in
  // world space — transforming its axis-aligned box by a rotation would inflate it — and each
  // point is mapped back through aniso^-1 for the containment test.
  //
  // The bbox condition keeps the mesh boundary untouched when bbox excludes the
  // boundary with a margin of at least one face. Snapping affects the boundary only by
  // moving a boundary vertex or subdividing a boundary edge, and the re-triangulation
  // of a patch keeps that patch's three original edges as boundary constraints. The
  // moved vertex and the subdivided edge both belong to the face whose interior
  // contains the projection, so they lie within one face of it. The projection is
  // inside bbox, which holds no boundary, so the touched feature is interior. (In the
  // isosurface pipeline bbox is first_extended_bbox: Lattice::cluster_vertices
  // guarantees no boundary vertex inside it, and the boundary lies a further lattice
  // layer out.)
  Snapper(const Points3& vertices, Faces faces, const geometry::Bbox3& bbox, double max_distance,
          const Mat3& aniso = Mat3::Identity())
      : mesh_(geometry::transform_points<3>(aniso, vertices), std::move(faces)),
        bbox_(bbox),
        to_iso_(aniso),
        to_world_(aniso.inverse()),
        max_distance_(max_distance),
        positions_(mesh_.vertices().rowwise().begin(), mesh_.vertices().rowwise().end()),
        moved_(mesh_.num_vertices(), false) {}

  // Must be called only once. points are in world space; the result is too. tolerances, if
  // non-empty, gives a per-point snapping tolerance, the distance the surface may stay from the
  // point. It is used twice: a point the partially snapped mesh already passes within its
  // tolerance of is skipped (snapping it would barely move the surface and only over-subdivide
  // the patch), and afterwards an inserted vertex whose removal keeps the surface within its
  // tolerance is dropped (see thin_inserted), so a densely sampled polyline does not over-
  // triangulate the surface. An empty vector means zero (snap every point in range, thin nothing).
  Mesh snap(const Points3& points, const VecX& tolerances = VecX()) {
    if (tolerances.size() != 0 && tolerances.size() != points.rows()) {
      throw std::invalid_argument("tolerances must be empty or have one entry per point");
    }
    auto iso_points = geometry::transform_points<3>(to_iso_, points);
    auto candidates = build_candidates(iso_points, tolerances);
    std::vector<bool> placed(candidates.size(), false);

    // Pass 1: vertex moves only. A candidate moves its nearest vertex only while that vertex
    // is nearer than any of its edges (the leading run of vertex simplices in the cascade).
    // Doing all vertex moves before any edge/face snap lets an original vertex absorb its
    // snap point first; otherwise an edge/face snap would subdivide the patch around it into
    // slivers, and the later move would fold those slivers and be rejected.
    for (std::size_t i = 0; i < candidates.size(); i++) {
      const auto& cand = candidates.at(i);
      if (already_satisfied(cand)) {
        placed.at(i) = true;
        stats_.satisfied++;
        continue;
      }
      for (auto code : cand.order) {
        if (index_of(code) >= index_of(Simplex::kEdge12)) {
          break;  // an edge or the face is nearer than the remaining vertices: defer to pass 2
        }
        if (try_place(cand, code)) {
          placed.at(i) = true;
          break;
        }
      }
    }

    // Pass 2: the full cascade (edge and face snaps) for everything still unplaced.
    for (std::size_t i = 0; i < candidates.size(); i++) {
      if (placed.at(i)) {
        continue;
      }
      const auto& cand = candidates.at(i);
      if (already_satisfied(cand)) {
        stats_.satisfied++;
        continue;
      }
      bool ok = false;
      for (auto code : cand.order) {
        if (try_place(cand, code)) {
          ok = true;
          break;
        }
      }
      if (!ok) {
        stats_.dropped++;
      }
    }

    // Drop inserted vertices a densely sampled polyline leaves behind that the surface no longer
    // needs (within each point's snapping tolerance), which would otherwise over-triangulate it
    // and block the edge-flip smoother. A no-op when no tolerances were given.
    thin_inserted();
    return emit();
  }

  const Stats& stats() const { return stats_; }

 private:
  // -- Classification (phase 1): project each point and rank the seven simplex
  // centroids of its face by distance to the projection (a Voronoi classification),
  // then order the candidates by increasing tangential offset (see the sort below).
  std::vector<Candidate> build_candidates(const Points3& points, const VecX& tolerances) {
    const auto& V = mesh_.vertices();
    const auto& F = mesh_.faces();

    std::vector<Candidate> candidates;
    candidates.reserve(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      Point3 p = points.row(i);

      int fi{};
      Point3 q;
      auto d2 = mesh_.nearest_face(p, fi, q);

      if (d2 > max_distance_ * max_distance_) {
        stats_.skipped++;
        continue;
      }
      Point3 p_world = p * to_world_.transpose();
      Point3 q_world = q * to_world_.transpose();
      if (!bbox_.contains(p_world) || !bbox_.contains(q_world)) {
        stats_.skipped++;
        continue;
      }

      auto f = F.row(fi);
      Point3 a = V.row(f(0));
      Point3 b = V.row(f(1));
      Point3 c = V.row(f(2));

      Vector3 l;
      igl::barycentric_coordinates(q, a, b, c, l);

      // The centroid of each simplex, indexed by Simplex (vertices, edge midpoints, face).
      std::array<Point3, 7> sites{
          a, b, c, 0.5 * (b + c), 0.5 * (c + a), 0.5 * (a + b), (a + b + c) / 3.0};
      std::array<Simplex, 7> order{Simplex::kVertex0, Simplex::kVertex1, Simplex::kVertex2,
                                   Simplex::kEdge12,  Simplex::kEdge20,  Simplex::kEdge01,
                                   Simplex::kFace};
      std::ranges::sort(order, [&](Simplex s, Simplex t) {
        return (q - sites.at(index_of(s))).squaredNorm() <
               (q - sites.at(index_of(t))).squaredNorm();
      });

      auto tang = (q - sites.at(index_of(order.front()))).squaredNorm();
      candidates.push_back({.p = p,
                            .d2 = d2,
                            .tang = tang,
                            .min_distance = tolerances.size() == 0 ? 0.0 : tolerances(i),
                            .face = fi,
                            .fv = {f(0), f(1), f(2)},
                            .l = {l(0), l(1), l(2)},
                            .order = order});
    }

    // Order by increasing tangential offset — the in-surface distance from a point's
    // projection to the feature it snaps to. A vertex move's distortion is its tangential
    // component (a perpendicular move only raises or lowers the height field and leaves the
    // patch shape unchanged), so processing the least-tangential snaps first lets each shared
    // vertex be claimed by its least-distorting candidate, independent of how far off the
    // surface the points lie.
    std::ranges::sort(candidates, [](const Candidate& x, const Candidate& y) {
      return std::make_tuple(x.tang, x.p(0), x.p(1), x.p(2)) <
             std::make_tuple(y.tang, y.p(0), y.p(1), y.p(2));
    });
    return candidates;
  }

  // -- Greedy placement. Returns true if the candidate was accepted at this simplex.
  bool try_place(const Candidate& cand, Simplex s) {
    switch (s) {
      case Simplex::kVertex0:
      case Simplex::kVertex1:
      case Simplex::kVertex2:
        return try_vertex(cand, cand.fv.at(index_of(s)));
      case Simplex::kEdge12:
      case Simplex::kEdge20:
      case Simplex::kEdge01:
        return try_edge(cand, index_of(s) - index_of(Simplex::kEdge12));
      case Simplex::kFace:
        return try_interior(cand);
    }
    return false;  // unreachable; all simplices are handled above
  }

  // Moving a shared vertex changes no connectivity, only the emitted position. Accept the
  // move unless an incident triangle becomes degenerate, overlaps a non-adjacent triangle in
  // its neighborhood, or pierces a distant face. (A fold that inverts a triangle is a
  // self-overlap and is caught by causes_overlap.)
  bool try_vertex(const Candidate& cand, Index v) {
    if (moved_.at(v)) {
      return false;
    }

    Point3 original = positions_.at(v);  // for revert
    positions_.at(v) = cand.p;           // tentative
    std::unordered_map<Index, std::vector<Face>> changed;
    for (auto fi : mesh_.vertex_faces(v)) {
      changed[fi] = patch_faces(fi);
    }
    if (causes_overlap(changed) || creates_degenerate(changed)) {
      positions_.at(v) = original;
      return false;
    }
    if (pierces(changed)) {
      stats_.pierce_rejections++;
      positions_.at(v) = original;
      return false;
    }

    moved_.at(v) = true;
    stats_.moved_vertices++;
    return true;
  }

  // Adding a vertex on the edge subdivides both incident patches. Accept if neither
  // re-triangulation intersects its neighborhood. `i` is the local index of the vertex
  // opposite the edge being snapped to; j and k are the edge's two endpoints.
  bool try_edge(const Candidate& cand, int i) {
    auto j = (i + 1) % 3;
    auto k = (i + 2) % 3;
    if (!(cand.l.at(j) + cand.l.at(k) > 0.0)) {
      return false;
    }
    auto vj = cand.fv.at(j);
    auto vk = cand.fv.at(k);
    auto t = cand.l.at(k) / (cand.l.at(j) + cand.l.at(k));
    if (vj > vk) {
      std::swap(vj, vk);
      t = 1.0 - t;
    }
    Edge e{.a = vj, .b = vk};
    const auto& incident_faces = mesh_.edge_faces(e);

    auto id = static_cast<Index>(positions_.size());
    positions_.push_back(cand.p);
    auto& chain = edge_chains_[e];
    chain.insert(std::ranges::lower_bound(chain, t, {}, &EdgeVertex::t), {.t = t, .id = id});

    std::unordered_map<Index, std::vector<Face>> changed;
    bool simple = true;
    for (auto fi : incident_faces) {
      bool patch_simple = true;
      changed[fi] = triangulate_patch(fi, &patch_simple);
      simple = simple && patch_simple;
    }
    auto revert = [&] {
      std::erase_if(chain, [id](const EdgeVertex& x) { return x.id == id; });
      if (chain.empty()) {
        edge_chains_.erase(e);
      }
      positions_.pop_back();
    };
    if (!simple || causes_overlap(changed) || creates_degenerate(changed)) {
      revert();
      return false;
    }
    if (pierces(changed)) {
      stats_.pierce_rejections++;
      revert();
      return false;
    }

    for (auto fi : incident_faces) {
      patch_faces_cache_[fi] = changed.at(fi);
    }
    inserted_tol_[id] = cand.min_distance;
    stats_.inserted_on_edges++;
    return true;
  }

  // Adding a vertex interior to a patch changes only that patch and never its boundary,
  // so it cannot overlap a neighbor in the height-field sense — but an off-surface
  // interior vertex can still pierce a distant sheet, so the global check applies. It is
  // also rejected if the triangulation drops it (the point fell outside the patch's
  // subdivided polygon or onto a constraint).
  bool try_interior(const Candidate& cand) {
    auto fi = cand.face;
    auto id = static_cast<Index>(positions_.size());
    positions_.push_back(cand.p);
    auto& interior = face_interior_[fi];
    interior.push_back(id);

    bool simple = true;
    auto faces = triangulate_patch(fi, &simple);
    bool used = false;
    for (const auto& face : faces) {
      for (auto x : face) {
        if (x == id) {
          used = true;
        }
      }
    }
    std::unordered_map<Index, std::vector<Face>> changed{{fi, faces}};
    auto revert = [&] {
      interior.pop_back();
      if (interior.empty()) {
        face_interior_.erase(fi);
      }
      positions_.pop_back();
    };
    if (!simple || !used || causes_overlap(changed) || creates_degenerate(changed)) {
      revert();
      return false;
    }
    if (pierces(changed)) {
      stats_.pierce_rejections++;
      revert();
      return false;
    }

    patch_faces_cache_[fi] = std::move(faces);
    inserted_tol_[id] = cand.min_distance;
    stats_.inserted_in_faces++;
    return true;
  }

  // -- Validity checks.

  // Whether any triangle of the changed patches is degenerate (near-zero area) when emitted
  // in 3D. A patch is triangulated in its 2D frame, but its vertices are emitted at their
  // off-surface 3D positions; a triangle valid in the frame can still be collinear in 3D (a
  // moved vertex nearly in line with its edge's chain vertices), which the 2D checks miss.
  bool creates_degenerate(const std::unordered_map<Index, std::vector<Face>>& changed) {
    const auto& V = mesh_.vertices();
    const auto& F = mesh_.faces();
    for (const auto& [fi, faces] : changed) {
      Point3 a = V.row(F(fi, 0));
      Point3 b = V.row(F(fi, 1));
      Point3 c = V.row(F(fi, 2));
      auto scale = (b - a).cross(c - a).norm();  // twice the original face's area
      for (const auto& face : faces) {
        Vector3 e1 = pos(face[1]) - pos(face[0]);
        Vector3 e2 = pos(face[2]) - pos(face[0]);
        if (e1.cross(e2).norm() <= 1e-9 * scale) {
          return true;
        }
      }
    }
    return false;
  }

  // Whether any triangle of the changed patches overlaps a non-adjacent triangle in its
  // neighborhood (the changed patches and their vertex one-ring). The test is done in
  // the changed patch's own 2D frame: a self-intersection-free snapped surface is a
  // height field over each patch, so two triangles overlapping when projected into the
  // frame (and z-separated or not) is exactly the failure to avoid. Working in the
  // frame also sidesteps the unreliable coplanar branch of a 3D triangle test. Edge-adjacent
  // triangles are skipped (they always touch along the shared edge); see tris_overlap_2d.
  bool causes_overlap(const std::unordered_map<Index, std::vector<Face>>& changed) {
    std::vector<Index> changed_ids;
    changed_ids.reserve(changed.size());
    for (const auto& [fi, faces] : changed) {
      changed_ids.push_back(fi);
    }

    std::vector<Face> neighborhood;
    for (const auto& [fi, faces] : changed) {
      neighborhood.insert(neighborhood.end(), faces.begin(), faces.end());
    }
    for (auto fi : one_ring(changed_ids)) {
      const auto& faces = patch_faces(fi);
      neighborhood.insert(neighborhood.end(), faces.begin(), faces.end());
    }

    for (const auto& [fi, faces] : changed) {
      for (const auto& a : faces) {
        for (const auto& b : neighborhood) {
          if (tris_overlap_2d(fi, a, b)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  // Whether triangles a and b overlap when projected into patch fi's frame, ignoring
  // pairs that share a vertex or are degenerate. Uses the separating-axis test, which is
  // exact for convex shapes in 2D.
  bool tris_overlap_2d(Index fi, const Face& a, const Face& b) {
    // Skip only edge-adjacent (or identical) pairs: they share an edge and always touch.
    // A pair sharing a single vertex is kept — two patches folding back onto each other
    // meet only at a vertex, and the separating-axis test below tells a real overlap (a
    // doubled surface) from a mere vertex touch (a valid crease).
    if (shared_vertices(a, b) >= 2) {
      return false;
    }
    std::array<Point2, 3> pa{mesh_.project(fi, pos(a[0])), mesh_.project(fi, pos(a[1])),
                             mesh_.project(fi, pos(a[2]))};
    std::array<Point2, 3> pb{mesh_.project(fi, pos(b[0])), mesh_.project(fi, pos(b[1])),
                             mesh_.project(fi, pos(b[2]))};
    // The slack (a real distance in the frame) lets a bare vertex/edge touch read as not
    // overlapping; kept tiny so a true doubled-surface overlap is still caught.
    return triangles_overlap_2d(pa, pb, 1e-9);
  }

  // The number of vertices triangles a and b share (0..3).
  static int shared_vertices(const Face& a, const Face& b) {
    int n = 0;
    for (auto u : a) {
      for (auto w : b) {
        if (u == w) {
          n++;
        }
      }
    }
    return n;
  }

  // The faces sharing a vertex with any of the given patches, excluding the patches
  // themselves.
  std::vector<Index> one_ring(const std::vector<Index>& patches) {
    const auto& F = mesh_.faces();
    std::unordered_set<Index> patch_set(patches.begin(), patches.end());
    std::unordered_set<Index> ring;
    for (auto fi : patches) {
      for (auto k = 0; k < 3; k++) {
        for (auto nf : mesh_.vertex_faces(F(fi, k))) {
          if (!patch_set.contains(nf)) {
            ring.insert(nf);
          }
        }
      }
    }
    return {ring.begin(), ring.end()};
  }

  // Whether any new triangle of the changed patches transversally intersects a face that
  // does not share a vertex with it — a self-intersection between geometrically near but
  // topologically distant parts of the surface, which the local height-field test cannot
  // see (and which an off-surface interior vertex can also cause). Original faces near
  // the new triangles are found by descending the AABB tree; their current triangulations
  // are tested with a 3D triangle-triangle intersection, skipping (near-)coplanar pairs
  // (handled by causes_overlap) and shared-vertex pairs.
  bool pierces(const std::unordered_map<Index, std::vector<Face>>& changed) {
    std::unordered_set<Index> changed_ids;
    std::vector<Face> changed_faces;
    for (const auto& [fi, faces] : changed) {
      changed_ids.insert(fi);
      changed_faces.insert(changed_faces.end(), faces.begin(), faces.end());
    }

    // A candidate face's snapped triangulation can be up to max_distance from its
    // original, and the new triangle up to max_distance from the original it is near, so
    // their original faces can be up to 2 * max_distance apart.
    auto margin = 2.0 * max_distance_;
    std::unordered_set<Index> candidates;
    for (const auto& t : changed_faces) {
      Point3 a = pos(t[0]);
      Point3 b = pos(t[1]);
      Point3 c = pos(t[2]);
      Point3 lo = (a.cwiseMin(b).cwiseMin(c).array() - margin).matrix();
      Point3 hi = (a.cwiseMax(b).cwiseMax(c).array() + margin).matrix();
      mesh_.faces_near(Eigen::AlignedBox3d(lo.transpose(), hi.transpose()), changed_ids,
                       candidates);
    }

    // Test the new triangles against the candidates' current triangulations, and against
    // one another (the changed patches may pierce each other).
    std::vector<Face> others = changed_faces;
    for (auto fj : candidates) {
      const auto& faces = patch_faces(fj);
      others.insert(others.end(), faces.begin(), faces.end());
    }
    for (const auto& a : changed_faces) {
      for (const auto& b : others) {
        if (intersects(a, b)) {
          return true;
        }
      }
    }
    return false;
  }

  // Whether disjoint triangles a and b intersect. This is the global piercing test: only
  // faces that share no vertex matter here, because a fold between faces that share a
  // vertex is a local (one-ring) event the height-field test in causes_overlap already
  // covers. With the pair disjoint, the 3D triangle-crossing test (which reports a bare
  // touch as an overlap) is exact, including for coplanar pairs.
  bool intersects(const Face& a, const Face& b) const {
    if (shared_vertices(a, b) != 0) {
      return false;
    }
    Eigen::Vector3d a0 = pos(a[0]).transpose();
    Eigen::Vector3d a1 = pos(a[1]).transpose();
    Eigen::Vector3d a2 = pos(a[2]).transpose();
    Eigen::Vector3d b0 = pos(b[0]).transpose();
    Eigen::Vector3d b1 = pos(b[1]).transpose();
    Eigen::Vector3d b2 = pos(b[2]).transpose();

    Eigen::Vector3d amin = a0.cwiseMin(a1).cwiseMin(a2);
    Eigen::Vector3d amax = a0.cwiseMax(a1).cwiseMax(a2);
    Eigen::Vector3d bmin = b0.cwiseMin(b1).cwiseMin(b2);
    Eigen::Vector3d bmax = b0.cwiseMax(b1).cwiseMax(b2);
    if ((amax.array() < bmin.array()).any() || (bmax.array() < amin.array()).any()) {
      return false;
    }

    return igl::tri_tri_overlap_test_3d(a0, a1, a2, b0, b1, b2);
  }

  // -- Geometry.

  // The current 3D position of a vertex: an original vertex's snapped target if it was
  // moved, otherwise its original position; or an inserted vertex's off-surface point.
  const Point3& pos(Index v) const { return positions_.at(v); }

  // The squared distance from p to triangle (a, b, c) (closest point on the triangle).
  static double point_tri_dist2(const Point3& p, const Point3& a, const Point3& b,
                                const Point3& c) {
    Vector3 ab = b - a;
    Vector3 ac = c - a;
    Vector3 ap = p - a;
    double d1 = ab.dot(ap);
    double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
      return ap.squaredNorm();
    }
    Vector3 bp = p - b;
    double d3 = ab.dot(bp);
    double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) {
      return bp.squaredNorm();
    }
    Vector3 cp = p - c;
    double d5 = ab.dot(cp);
    double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) {
      return cp.squaredNorm();
    }
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
      double v = d1 / (d1 - d3);
      return (ap - v * ab).squaredNorm();
    }
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
      double w = d2 / (d2 - d6);
      return (ap - w * ac).squaredNorm();
    }
    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
      double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return (p - (b + w * (c - b))).squaredNorm();
    }
    double denom = 1.0 / (va + vb + vc);
    double v = vb * denom;
    double w = vc * denom;
    return (p - (a + ab * v + ac * w)).squaredNorm();
  }

  // Whether the current (partially snapped) mesh already passes within the candidate's tolerance
  // (its per-point min_distance) of the point, in which case snapping it would barely move the
  // surface and only over-subdivide the patch, so it is skipped. Checks the point's projected
  // patch and the patches across its edges (the point may have classified onto an edge).
  bool already_satisfied(const Candidate& cand) {
    if (!(cand.min_distance > 0.0)) {
      return false;
    }
    double tol2 = cand.min_distance * cand.min_distance;
    const auto& F = mesh_.faces();
    auto nearest_in_patch = [&](Index fi) {
      double best = std::numeric_limits<double>::infinity();
      for (const auto& f : patch_faces(fi)) {
        best = std::min(best, point_tri_dist2(cand.p, pos(f[0]), pos(f[1]), pos(f[2])));
      }
      return best;
    };
    if (nearest_in_patch(cand.face) < tol2) {
      return true;
    }
    for (auto k = 0; k < 3; k++) {
      auto e = make_edge(F(cand.face, k), F(cand.face, (k + 1) % 3));
      for (auto fj : mesh_.edge_faces(e)) {
        if (fj != cand.face && nearest_in_patch(fj) < tol2) {
          return true;
        }
      }
    }
    return false;
  }

  // The constrained Delaunay triangulation of a patch over its committed edge chains and
  // interior vertices, as triples of vertex ids.
  std::vector<Face> triangulate_patch(Index fi, bool* simple = nullptr) {
    if (simple != nullptr) {
      *simple = true;
    }
    const auto& F = mesh_.faces();
    std::array<Index, 3> vertices{F(fi, 0), F(fi, 1), F(fi, 2)};

    auto has_chain = [&](Index u, Index w) { return edge_chains_.contains(make_edge(u, w)); };
    if (!face_interior_.contains(fi) && !has_chain(vertices[0], vertices[1]) &&
        !has_chain(vertices[1], vertices[2]) && !has_chain(vertices[2], vertices[0])) {
      return {vertices};
    }

    std::vector<Point2> boundary;
    std::vector<Index> boundary_ids;
    // The original edge(s) each boundary vertex lies on, so the triangulation never cuts a
    // diagonal along a subdivided edge (see Triangulation). Patch edge k joins vertices k
    // and (k + 1) % 3; vertex k therefore lies on edges k and (k + 2) % 3.
    std::vector<std::array<int, 2>> boundary_edges;
    auto add_vertex = [&](int k) {
      boundary_ids.push_back(vertices.at(k));
      // Project the vertex's *emitted* position (a moved vertex's snapped target), as the
      // chains and interior vertices do, so the 2D triangulation matches the 3D mesh that is
      // emitted. Using the original position here would leave a moved vertex and its edge's
      // near-collinear chain vertices forming a triangle that is valid in the frame but
      // degenerate in 3D.
      boundary.push_back(mesh_.project(fi, pos(vertices.at(k))));
      boundary_edges.push_back({k, (k + 2) % 3});
    };
    auto add_chain = [&](int edge) {
      auto from = vertices.at(edge);
      auto to = vertices.at((edge + 1) % 3);
      auto it = edge_chains_.find(make_edge(from, to));
      if (it == edge_chains_.end()) {
        return;
      }
      const auto& chain = it->second;  // stored by t from the smaller id to the larger
      auto append = [&](Index v) {
        boundary_ids.push_back(v);
        boundary.push_back(mesh_.project(fi, pos(v)));
        boundary_edges.push_back({edge, -1});
      };
      if (from < to) {
        for (const auto& x : chain) {
          append(x.id);
        }
      } else {
        for (auto i = static_cast<Index>(chain.size()) - 1; i >= 0; i--) {
          append(chain.at(i).id);
        }
      }
    };
    add_vertex(0);
    add_chain(0);
    add_vertex(1);
    add_chain(1);
    add_vertex(2);
    add_chain(2);

    std::vector<Point2> interior;
    std::vector<Index> interior_ids;
    if (auto it = face_interior_.find(fi); it != face_interior_.end()) {
      for (auto v : it->second) {
        interior_ids.push_back(v);
        interior.push_back(mesh_.project(fi, pos(v)));
      }
    }

    auto nb = static_cast<Index>(boundary_ids.size());
    Triangulation triangulation(boundary, interior, std::move(boundary_edges));
    if (simple != nullptr) {
      *simple = triangulation.simple();
    }
    std::vector<Face> faces;
    for (const auto& t : triangulation.faces()) {
      auto map = [&](Index i) { return i < nb ? boundary_ids.at(i) : interior_ids.at(i - nb); };
      faces.push_back({map(t[0]), map(t[1]), map(t[2])});
    }
    return faces;
  }

  // The cached triangulation of a patch (computed on first use).
  const std::vector<Face>& patch_faces(Index fi) {
    auto it = patch_faces_cache_.find(fi);
    if (it == patch_faces_cache_.end()) {
      it = patch_faces_cache_.emplace(fi, triangulate_patch(fi)).first;
    }
    return it->second;
  }

  // The squared distance from p to the nearest of the given triangles (emitted 3D positions).
  double dist2_to_faces(const Point3& p, const std::vector<Face>& faces) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& f : faces) {
      best = std::min(best, point_tri_dist2(p, pos(f[0]), pos(f[1]), pos(f[2])));
    }
    return best;
  }

  // Try to drop interior vertex `id` from patch `fi`: removing it re-triangulates the one patch.
  // Committed only if the new patch passes within tol of the dropped point and leaves the mesh
  // free of self-intersection (the insertion guards). Returns whether it was dropped.
  bool try_drop_interior(Index fi, Index id, double tol) {
    auto it = face_interior_.find(fi);
    if (it == face_interior_.end()) {
      return false;
    }
    auto& interior = it->second;
    auto at = std::ranges::find(interior, id);
    if (at == interior.end()) {
      return false;
    }
    interior.erase(at);
    auto faces = triangulate_patch(fi);
    std::unordered_map<Index, std::vector<Face>> changed{{fi, faces}};
    if (dist2_to_faces(pos(id), faces) <= tol * tol && !causes_overlap(changed) &&
        !creates_degenerate(changed) && !pierces(changed)) {
      patch_faces_cache_[fi] = std::move(faces);
      if (interior.empty()) {
        face_interior_.erase(fi);
      }
      stats_.thinned_in_faces++;
      return true;
    }
    interior.push_back(id);  // revert (interior order does not affect the triangulation)
    return false;
  }

  // Try to drop edge vertex `id` from edge `e`'s chain: removing it re-triangulates both incident
  // patches. Same accept test as try_drop_interior, against both patches' new triangulation.
  bool try_drop_edge(const Edge& e, Index id, double tol) {
    auto it = edge_chains_.find(e);
    if (it == edge_chains_.end()) {
      return false;
    }
    auto& chain = it->second;
    auto at = std::ranges::find_if(chain, [id](const EdgeVertex& x) { return x.id == id; });
    if (at == chain.end()) {
      return false;
    }
    EdgeVertex removed = *at;
    auto pos_in_chain = at - chain.begin();
    chain.erase(at);
    const auto& incident = mesh_.edge_faces(e);
    std::unordered_map<Index, std::vector<Face>> changed;
    std::vector<Face> all;
    for (auto fi : incident) {
      auto faces = triangulate_patch(fi);
      all.insert(all.end(), faces.begin(), faces.end());
      changed[fi] = std::move(faces);
    }
    if (dist2_to_faces(pos(id), all) <= tol * tol && !causes_overlap(changed) &&
        !creates_degenerate(changed) && !pierces(changed)) {
      for (auto fi : incident) {
        patch_faces_cache_[fi] = changed.at(fi);
      }
      if (chain.empty()) {
        edge_chains_.erase(e);
      }
      stats_.thinned_on_edges++;
      return true;
    }
    chain.insert(chain.begin() + pos_in_chain, removed);  // revert
    return false;
  }

  // Drop inserted vertices a densely sampled polyline leaves behind that the surface no longer
  // needs: a vertex whose removal keeps the surface within its own snapping tolerance and free of
  // self-intersection is redundant (the surrounding snaps already pin the surface through it). Such
  // runs over-triangulate the surface -- e.g. into vertical fins -- and block the edge-flip
  // smoother. Unlike a per-chain collinearity test this also reaches face-interior vertices and
  // single-vertex chains, which is most of a polyline that crosses the mesh transversally. Greedy
  // and iterated to a fixpoint, since one drop can make a neighbour removable; a zero-tolerance
  // vertex is never dropped, so this is a no-op when no tolerances were given. The dropped vertices
  // become unreferenced and emit() compacts them away.
  void thin_inserted() {
    bool any = true;
    while (any) {
      any = false;
      std::vector<Index> faces;
      faces.reserve(face_interior_.size());
      for (const auto& [fi, ids] : face_interior_) {
        faces.push_back(fi);
      }
      for (auto fi : faces) {
        auto it = face_interior_.find(fi);
        if (it == face_interior_.end()) {
          continue;
        }
        for (auto id : std::vector<Index>(it->second)) {
          double tol = inserted_tol_.at(id);
          if (tol > 0.0 && try_drop_interior(fi, id, tol)) {
            any = true;
          }
        }
      }
      std::vector<Edge> edges;
      edges.reserve(edge_chains_.size());
      for (const auto& [e, chain] : edge_chains_) {
        edges.push_back(e);
      }
      for (const auto& e : edges) {
        auto it = edge_chains_.find(e);
        if (it == edge_chains_.end()) {
          continue;
        }
        std::vector<Index> ids;
        ids.reserve(it->second.size());
        for (const auto& x : it->second) {
          ids.push_back(x.id);
        }
        for (auto id : ids) {
          double tol = inserted_tol_.at(id);
          if (tol > 0.0 && try_drop_edge(e, id, tol)) {
            any = true;
          }
        }
      }
    }
  }

  // Builds the snapped mesh from the accepted snaps.
  Mesh emit() {
    const auto& F = mesh_.faces();

    std::vector<Face> faces;
    for (Index fi = 0; fi < F.rows(); fi++) {
      for (const auto& face : patch_faces(fi)) {
        faces.push_back(face);
      }
    }

    // Drop vertices no face references (chain thinning orphans the inserted ones it removes),
    // keeping the rest in their original order so an un-thinned run emits unchanged.
    std::vector<bool> used(positions_.size(), false);
    for (const auto& face : faces) {
      for (auto v : face) {
        used.at(v) = true;
      }
    }
    std::vector<Index> remap(positions_.size(), -1);
    Index n = 0;
    for (std::size_t v = 0; v < positions_.size(); v++) {
      if (used.at(v)) {
        remap.at(v) = n++;
      }
    }

    Points3 vertices(n, 3);
    for (std::size_t v = 0; v < positions_.size(); v++) {
      if (used.at(v)) {
        vertices.row(remap.at(v)) = positions_.at(v);
      }
    }
    vertices = geometry::transform_points<3>(to_world_, vertices);  // back to world

    Faces f(static_cast<Index>(faces.size()), 3);
    for (Index i = 0; i < f.rows(); i++) {
      f(i, 0) = remap.at(faces.at(i)[0]);
      f(i, 1) = remap.at(faces.at(i)[1]);
      f(i, 2) = remap.at(faces.at(i)[2]);
    }
    return {std::move(vertices), std::move(f)};
  }

  OriginalMesh mesh_;
  geometry::Bbox3 bbox_;
  Mat3 to_iso_;    // world -> the lattice's isotropic frame, where snapping is done (= aniso)
  Mat3 to_world_;  // the isotropic frame -> world, for the bbox test and emit (= aniso^-1)
  double max_distance_;

  // The accumulated snap edits and their derived caches. positions_ is every vertex's current
  // position (original then inserted); moved_ flags which original vertices have been claimed
  // by a move (so a second snap point cannot move the same one).
  std::vector<Point3> positions_;
  std::vector<bool> moved_;
  std::unordered_map<Edge, std::vector<EdgeVertex>, EdgeHash> edge_chains_;
  std::unordered_map<Index, std::vector<Index>> face_interior_;
  std::unordered_map<Index, std::vector<Face>> patch_faces_cache_;
  std::unordered_map<Index, double> inserted_tol_;  // each inserted vertex's snapping tolerance

  Stats stats_;
};

}  // namespace polatory::isosurface::snapper
