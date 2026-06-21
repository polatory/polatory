#pragma once

#include <igl/barycentric_coordinates.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
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
//    on its edges and interior — is then re-triangulated from scratch by a constrained Delaunay
//    triangulation (see Triangulation) over the vertices' *on-surface* positions, with the
//    subdivided original edges as boundary constraints. Deciding connectivity on the flat
//    surface, before the vertices are moved to their off-surface snap targets, keeps the
//    triangulation valid and consistently wound no matter how steeply a feature is snapped.
//
//  - Accept or drop. Keep the snap only if the emitted (moved) mesh does not self-intersect: the
//    affected patches are tested against every nearby face and rejected if any pair actually
//    crosses. A sharp crease that merely folds over a patch's plane but meets it only along an
//    edge is allowed — the guarantee is no self-intersection, not a single-valued height field.
//    A rejected vertex or edge snap cascades to its next-nearest simplex (ultimately a face snap,
//    which perturbs only its own patch); a point that cannot be placed anywhere is dropped.
//
// Processing order. Points are taken by increasing distance to the mesh — the perpendicular move
// a snap makes. Taking the least-distorting snaps first lets each shared feature be claimed by the
// candidate that moves it least; ordering instead by the in-surface (tangential) offset let a far
// point win a contended vertex over a near one, folding the patch into an overhang.
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
    Point3 p;                   // The point in the isotropic frame, where snapping is measured.
    Point3 p_world;             // Its exact world position, emitted as-is (no aniso round-trip).
    Point3 q;                   // The projection of p onto the mesh (closest point).
    double d2{};                // The squared distance from p to the mesh.
    double min_distance{};      // Skip if the mesh passes within this of p (0 = only when on it).
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
        flat_(mesh_.vertices().rowwise().begin(), mesh_.vertices().rowwise().end()),
        world_(vertices.rowwise().begin(), vertices.rowwise().end()),
        moved_(mesh_.num_vertices(), false) {}

  // Must be called only once. points are in world space; the result is too. tolerances, if
  // non-empty, gives a per-point snapping tolerance, the distance the surface may stay from the
  // point: a point the partially snapped mesh already passes within its tolerance of is skipped,
  // since snapping it would barely move the surface and only over-subdivide the patch. (Dropping
  // the redundant vertices a dense polyline still leaves is done afterwards by the cross-pass
  // Thinner; see snapper/thinner.hpp.) An empty vector means zero (snap every point in range).
  Mesh snap(const Points3& points, const VecX& tolerances = VecX()) {
    if (tolerances.size() != 0 && tolerances.size() != points.rows()) {
      throw std::invalid_argument("tolerances must be empty or have one entry per point");
    }
    auto candidates = build_candidates(points, tolerances);
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
      // A point nearest a vertex that the cascade's single face could not place gets one more
      // chance: insertion on any edge or face containing that vertex across its whole umbrella.
      if (!ok && index_of(cand.order.front()) <= index_of(Simplex::kVertex2)) {
        ok = try_vertex_umbrella(cand, cand.fv.at(index_of(cand.order.front())));
      }
      if (!ok) {
        stats_.dropped++;
      }
    }

    return emit();
  }

  const Stats& stats() const { return stats_; }

 private:
  // Whether the current (partially snapped) mesh already passes within the candidate's tolerance
  // (its per-point min_distance) of the point, in which case snapping it would barely move the
  // surface and only over-subdivide the patch, so it is skipped. The bound is inclusive, so a
  // point the surface already passes exactly through is satisfied even at zero tolerance (a vertex
  // already snapped to it stays put on the next pass). Checks the point's projected patch and the
  // patches across its edges (the point may have classified onto an edge).
  bool already_satisfied(const Candidate& cand) {
    double tol2 = cand.min_distance * cand.min_distance;
    const auto& F = mesh_.faces();
    auto nearest_in_patch = [&](Index fi) {
      double best = std::numeric_limits<double>::infinity();
      for (const auto& f : patch_faces(fi)) {
        best = std::min(best, point_triangle_dist2(cand.p, pos(f[0]), pos(f[1]), pos(f[2])));
      }
      return best;
    };
    if (nearest_in_patch(cand.face) <= tol2) {
      return true;
    }
    for (auto k = 0; k < 3; k++) {
      Edge e{F(cand.face, k), F(cand.face, (k + 1) % 3)};
      for (auto fj : mesh_.edge_faces(e)) {
        if (fj != cand.face && nearest_in_patch(fj) <= tol2) {
          return true;
        }
      }
    }
    return false;
  }

  // Project each point and rank the seven simplex centroids of its face by distance to the
  // projection (a Voronoi classification), then order the candidates by increasing tangential
  // offset (see the sort below).
  std::vector<Candidate> build_candidates(const Points3& points, const VecX& tolerances) {
    const auto& V = mesh_.vertices();
    const auto& F = mesh_.faces();

    std::vector<Candidate> candidates;
    candidates.reserve(points.rows());
    for (Index i = 0; i < points.rows(); i++) {
      Point3 p_world = points.row(i);
      Point3 p = p_world * to_iso_.transpose();  // the isotropic frame, where snapping is measured

      int fi{};
      Point3 q;
      auto d2 = mesh_.nearest_face(p, fi, q);

      if (d2 > max_distance_ * max_distance_) {
        stats_.skipped++;
        continue;
      }
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

      candidates.push_back({.p = p,
                            .p_world = p_world,
                            .q = q,
                            .d2 = d2,
                            .min_distance = tolerances.size() == 0 ? 0.0 : tolerances(i),
                            .face = fi,
                            .fv = {f(0), f(1), f(2)},
                            .l = {l(0), l(1), l(2)},
                            .order = order});
    }

    // Order by increasing distance to the mesh — the perpendicular move a snap makes. Taking the
    // least-distorting snaps first lets each shared feature be claimed by the candidate that moves
    // it least; an earlier ordering by the in-surface (tangential) offset instead let a far point
    // win a contended vertex over a near one, folding the patch into an overhang.
    std::ranges::sort(candidates, [](const Candidate& x, const Candidate& y) {
      return std::make_tuple(x.d2, x.p(0), x.p(1), x.p(2)) <
             std::make_tuple(y.d2, y.p(0), y.p(1), y.p(2));
    });
    return candidates;
  }

  // Whether any triangle of the changed patches is degenerate (near-zero area) when emitted in 3D.
  // A patch is triangulated over its flat on-surface positions, but its vertices are emitted at
  // their moved 3D positions; a triangle valid on the flat surface can still be collinear in 3D (a
  // moved vertex nearly in line with its edge's chain vertices), which the flat triangulation
  // misses.
  bool creates_degenerate(const std::unordered_map<Index, std::vector<Face>>& changed) {
    for (const auto& [fi, faces] : changed) {
      auto scale = original_normal(fi).norm();  // twice the original face's area
      for (const auto& face : faces) {
        if (emitted_normal(face).norm() <= 1e-9 * scale) {
          return true;
        }
      }
    }
    return false;
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
        vertices.row(remap.at(v)) = world_.at(v);  // exact world position; no aniso round-trip
      }
    }

    Faces f(static_cast<Index>(faces.size()), 3);
    for (Index i = 0; i < f.rows(); i++) {
      f(i, 0) = remap.at(faces.at(i)[0]);
      f(i, 1) = remap.at(faces.at(i)[1]);
      f(i, 2) = remap.at(faces.at(i)[2]);
    }
    return {std::move(vertices), std::move(f)};
  }

  // The emitted normal of a face (over its vertices' current snapped positions).
  Vector3 emitted_normal(const Face& f) const { return normal(pos(f[0]), pos(f[1]), pos(f[2])); }

  // Inserts cand.p into face fi's interior; reverts unless the re-triangulation uses it (it
  // projects inside the patch) and stays free of self-intersection.
  bool insert_in_face(const Candidate& cand, Index fi) {
    auto id = static_cast<Index>(positions_.size());
    positions_.push_back(cand.p);
    world_.push_back(cand.p_world);
    flat_.push_back(cand.q);  // the snap point's on-surface projection
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
      world_.pop_back();
      flat_.pop_back();
    };
    if (!simple || !used || creates_degenerate(changed) || over_pulled(changed) || self_intersects(changed)) {
      revert();
      return false;
    }

    patch_faces_cache_[fi] = std::move(faces);
    stats_.inserted_in_faces++;
    return true;
  }

  // Inserts cand.p on edge e at parameter t from e's smaller-id endpoint, subdividing both its
  // incident patches; reverts unless the re-triangulations stay free of self-intersection.
  bool insert_on_edge(const Candidate& cand, const Edge& e, double t) {
    const auto& incident_faces = mesh_.edge_faces(e);

    auto id = static_cast<Index>(positions_.size());
    positions_.push_back(cand.p);
    world_.push_back(cand.p_world);
    flat_.push_back((1.0 - t) * flat_.at(e.a) + t * flat_.at(e.b));  // on the original edge
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
      world_.pop_back();
      flat_.pop_back();
    };
    if (!simple || creates_degenerate(changed) || over_pulled(changed) || self_intersects(changed)) {
      revert();
      return false;
    }

    for (auto fi : incident_faces) {
      patch_faces_cache_[fi] = changed.at(fi);
    }
    stats_.inserted_on_edges++;
    return true;
  }

  // The unnormalized normal of triangle (a, b, c); its length is twice the area.
  static Vector3 normal(const Point3& a, const Point3& b, const Point3& c) {
    return Vector3((b - a).cross(c - a));
  }

  // The original face fi's plane normal (over the unsnapped vertices).
  Vector3 original_normal(Index fi) const {
    const auto& V = mesh_.vertices();
    const auto& F = mesh_.faces();
    return normal(V.row(F(fi, 0)), V.row(F(fi, 1)), V.row(F(fi, 2)));
  }

  // Whether any emitted triangle is folded over: tilted past vertical from its original face's
  // plane, so its normal opposes the original's. A dense run of collinear contour points can
  // otherwise let a snap fold a thin sub-face back over the surface; rejecting that drops the point
  // or leaves it to a gentler simplex. A face tilted up to vertical (an acute feature) is kept --
  // a fin there is an under-resolution of the feature, for thinning or a finer mesh, not this guard.
  bool over_pulled(const std::unordered_map<Index, std::vector<Face>>& changed) {
    for (const auto& [fi, faces] : changed) {
      Vector3 o = original_normal(fi);
      for (const auto& face : faces) {
        if (emitted_normal(face).dot(o) < 0.0) {
          return true;
        }
      }
    }
    return false;
  }

  // The cached triangulation of a patch (computed on first use).
  const std::vector<Face>& patch_faces(Index fi) {
    auto it = patch_faces_cache_.find(fi);
    if (it == patch_faces_cache_.end()) {
      it = patch_faces_cache_.emplace(fi, triangulate_patch(fi)).first;
    }
    return it->second;
  }

  // The current 3D position of a vertex: an original vertex's snapped target if it was
  // moved, otherwise its original position; or an inserted vertex's off-surface point.
  const Point3& pos(Index v) const { return positions_.at(v); }

  // Whether any triangle of the changed patches actually intersects another face of the surface.
  // This is the snapper's one geometric acceptance test: patches are triangulated over their flat
  // (on-surface) positions, so the connectivity is always valid and consistently wound regardless
  // of how far the vertices are then snapped (see triangulate_patch); the only thing left to
  // forbid is an actual self-intersection of the emitted mesh. Each changed triangle is tested
  // against every face within 2 * max_distance (the farthest a snapped face can stray from the
  // original it might meet), found by descending the AABB tree.
  bool self_intersects(const std::unordered_map<Index, std::vector<Face>>& changed) {
    std::unordered_set<Index> changed_ids;
    std::vector<Face> changed_faces;
    for (const auto& [fi, faces] : changed) {
      changed_ids.insert(fi);
      changed_faces.insert(changed_faces.end(), faces.begin(), faces.end());
    }
    // A disjoint pair gets the robust exact crossing test; a vertex-sharing pair gets the crease-
    // aware test, which tells a true fold from the bare touch of a sharp crease that the exact test
    // would report as an overlap (see triangles_intersect).
    auto crosses = [&](const Face& a, const Face& b) {
      return triangles_intersect(pos(a[0]), pos(a[1]), pos(a[2]), pos(b[0]), pos(b[1]), pos(b[2]),
                                 shared_vertices(a, b));
    };
    // Each changed face is tested against the other changed faces and against the faces in *its
    // own* AABB-neighborhood (within 2 * max_distance, the farthest a snapped face can stray).
    // Querying per changed face, rather than pooling every changed face's neighborhood and testing
    // the full cross product, avoids checking a face against the many faces that are near a
    // different changed face but AABB-disjoint from this one -- they cannot cross it.
    auto margin = 2.0 * max_distance_;
    std::unordered_set<Index> candidates;
    for (const auto& a : changed_faces) {
      for (const auto& b : changed_faces) {
        if (crosses(a, b)) {
          return true;
        }
      }
      Point3 p0 = pos(a[0]);
      Point3 p1 = pos(a[1]);
      Point3 p2 = pos(a[2]);
      Point3 lo = (p0.cwiseMin(p1).cwiseMin(p2).array() - margin).matrix();
      Point3 hi = (p0.cwiseMax(p1).cwiseMax(p2).array() + margin).matrix();
      candidates.clear();
      mesh_.faces_near(Eigen::AlignedBox3d(lo.transpose(), hi.transpose()), changed_ids,
                       candidates);
      for (auto fj : candidates) {
        for (const auto& b : patch_faces(fj)) {
          if (crosses(a, b)) {
            return true;
          }
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

    auto has_chain = [&](const Edge& e) { return edge_chains_.contains(e); };
    if (!face_interior_.contains(fi) && !has_chain({vertices[0], vertices[1]}) &&
        !has_chain({vertices[1], vertices[2]}) && !has_chain({vertices[2], vertices[0]})) {
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
      // Project the vertex's *flat* (on-surface) position, not its snapped target, so the 2D
      // triangulation is over the unsnapped surface and is always a valid, non-folding,
      // consistently-wound polygon. The emitted mesh then moves to the snapped positions; a fold
      // or degeneracy of the moved mesh is caught by self_intersects and creates_degenerate.
      boundary.push_back(mesh_.project(fi, flat_.at(vertices.at(k))));
      boundary_edges.push_back({k, (k + 2) % 3});
    };
    auto add_chain = [&](int edge) {
      auto from = vertices.at(edge);
      auto to = vertices.at((edge + 1) % 3);
      auto it = edge_chains_.find({from, to});
      if (it == edge_chains_.end()) {
        return;
      }
      const auto& chain = it->second;  // stored by t from the smaller id to the larger
      auto append = [&](Index v) {
        boundary_ids.push_back(v);
        boundary.push_back(mesh_.project(fi, flat_.at(v)));
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
        interior.push_back(mesh_.project(fi, flat_.at(v)));
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
    return insert_on_edge(cand, Edge{vj, vk}, t);
  }

  // Adding a vertex interior to a patch changes only that patch, but its off-surface position can
  // still make the patch self-intersect a distant sheet, so the global check applies. It is also
  // rejected if the triangulation drops it (the point fell outside the patch's subdivided polygon
  // or onto a constraint).
  bool try_interior(const Candidate& cand) { return insert_in_face(cand, cand.face); }

  // Returns true if the candidate was accepted at this simplex.
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

  // Moving a shared vertex changes no connectivity, only the emitted position. Accept the move
  // unless an incident triangle becomes degenerate or self-intersects the surface.
  bool try_vertex(const Candidate& cand, Index v) {
    if (moved_.at(v)) {
      return false;
    }

    Point3 original = positions_.at(v);        // for revert
    Point3 original_world = world_.at(v);
    positions_.at(v) = cand.p;                 // tentative
    world_.at(v) = cand.p_world;
    std::unordered_map<Index, std::vector<Face>> changed;
    for (auto fi : mesh_.vertex_faces(v)) {
      changed[fi] = patch_faces(fi);
    }
    if (creates_degenerate(changed) || over_pulled(changed) || self_intersects(changed)) {
      positions_.at(v) = original;
      world_.at(v) = original_world;
      return false;
    }

    moved_.at(v) = true;
    stats_.moved_vertices++;
    return true;
  }

  // For a point whose nearest simplex is vertex v, try inserting it on any edge or face that
  // contains v across v's umbrella, nearest first. The cascade only saw the single AABB face,
  // so a spoke or face of another incident face may still take a point that face had to drop.
  bool try_vertex_umbrella(const Candidate& cand, Index v) {
    const auto& V = mesh_.vertices();
    const auto& F = mesh_.faces();

    // The edges of the cascade's own face that contain v were already tried; skip them.
    std::unordered_set<Edge, EdgeHash> queued;
    for (auto k = 0; k < 3; k++) {
      Index a = F(cand.face, k);
      Index b = F(cand.face, (k + 1) % 3);
      if (a == v || b == v) {
        queued.insert({a, b});
      }
    }

    struct Op {
      double dist2{};
      Index ea{-1};
      Index eb{-1};
      Index face{-1};
      double t{};
    };
    std::vector<Op> ops;
    for (auto fi : mesh_.vertex_faces(v)) {
      if (fi == cand.face) {
        continue;
      }
      Point3 a = V.row(F(fi, 0));
      Point3 b = V.row(F(fi, 1));
      Point3 c = V.row(F(fi, 2));
      Point3 centroid = a + ((b - a) + (c - a)) / 3.0;
      ops.push_back({.dist2 = (cand.q - centroid).squaredNorm(), .face = fi});
      for (auto k = 0; k < 3; k++) {
        Index x = F(fi, k);
        Index y = F(fi, (k + 1) % 3);
        if (x != v && y != v) {
          continue;  // not a spoke
        }
        Edge e{x, y};
        if (!queued.insert(e).second) {
          continue;  // already tried, or a spoke shared with another incident face
        }
        Point3 va = V.row(e.a);
        Point3 vb = V.row(e.b);
        Vector3 d = vb - va;
        auto t = (cand.q - va).dot(d) / d.squaredNorm();
        if (!(t > 0.0 && t < 1.0)) {
          continue;  // projects onto an endpoint
        }
        Point3 mid = va + 0.5 * d;
        ops.push_back({.dist2 = (cand.q - mid).squaredNorm(), .ea = e.a, .eb = e.b, .t = t});
      }
    }

    std::ranges::sort(ops, [](const Op& x, const Op& y) {
      return std::make_tuple(x.dist2, x.face, x.ea, x.eb) <
             std::make_tuple(y.dist2, y.face, y.ea, y.eb);
    });
    for (const auto& op : ops) {
      auto ok = op.face >= 0 ? insert_in_face(cand, op.face)
                             : insert_on_edge(cand, Edge{op.ea, op.eb}, op.t);
      if (ok) {
        return true;
      }
    }
    return false;
  }

  OriginalMesh mesh_;
  geometry::Bbox3 bbox_;
  Mat3 to_iso_;    // world -> the lattice's isotropic frame, where snapping is done (= aniso)
  Mat3 to_world_;  // the isotropic frame -> world, for the bbox test and emit (= aniso^-1)
  double max_distance_;

  // The accumulated snap edits and their derived caches. positions_ is every vertex's current
  // position in the isotropic frame (original then inserted; a move/insert sets the snapped
  // target), where all the geometry is measured. world_ is the same vertices' exact world
  // positions -- the original input vertices and, for a snap, the exact input snap point -- which
  // emit() outputs directly, so a snapped surface passes exactly through its points rather than
  // through aniso^-1 * aniso of them. flat_ is every vertex's on-surface position -- originals
  // unmoved, inserted vertices at their on-mesh projection -- which the triangulation projects so
  // connectivity is decided on the flat surface and never folds. moved_ flags which original
  // vertices have been claimed by a move (so a second snap point cannot move the same one).
  std::vector<Point3> positions_;
  std::vector<Point3> flat_;
  std::vector<Point3> world_;
  std::vector<bool> moved_;
  std::unordered_map<Edge, std::vector<EdgeVertex>, EdgeHash> edge_chains_;
  std::unordered_map<Index, std::vector<Index>> face_interior_;
  std::unordered_map<Index, std::vector<Face>> patch_faces_cache_;

  Stats stats_;
};

}  // namespace polatory::isosurface::snapper
