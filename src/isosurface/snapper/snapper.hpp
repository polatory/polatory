#pragma once

#include <igl/barycentric_coordinates.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <cstddef>
#include <format>
#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "abstract_mesh.hpp"
#include "spatial_grid.hpp"
#include "triangulation.hpp"
#include "utility.hpp"

namespace polatory::isosurface::snapper {

// Snaps a mesh to a subset of the given points without introducing self-intersection: each point is
// snapped or dropped, and the result provably has none. Three steps per point:
//
//  - Classify against the original mesh: the nearest simplex of its closest face (vertex, edge, or
//    interior) by which simplex centroid is nearest the projection.
//  - Snap: a vertex match moves that vertex; an edge match inserts a vertex on the shared
//    subdivided edge; a face match inserts one interior to a patch. Each affected patch is
//    re-triangulated by a constrained Delaunay triangulation over the *flat* on-surface positions
//    -- deciding connectivity before the vertices move keeps it valid and consistently wound
//    however steep the snap.
//  - Accept or drop: keep only if the moved mesh stays self-intersection-free (a crease folding to
//    a bare edge touch is allowed); else cascade to the next-nearest simplex, or drop.
//
// Points are processed by increasing distance to the mesh, so each shared feature is first claimed
// by the candidate that moves it least (a tangential order would fold a far point's patch into an
// overhang). A claimed vertex may still be re-moved to a farther point as long as the points it
// already snapped stay honored, letting it migrate out to a contour tip (see try_vertex). Boundary:
// a point snaps only if it and its projection lie in bbox; if bbox excludes
// the boundary, the boundary is provably untouched (see the constructor).
class Snapper {
  using Point2 = geometry::Point2;
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using Vector3 = geometry::Vector3;

  // A simplex of the projected face the point may snap to; the values double as indices into the
  // per-face site arrays (vertices 0..2, edges 3..5, face 6).
  enum class Simplex { kVertex0, kVertex1, kVertex2, kEdge12, kEdge20, kEdge01, kFace };

  static int index_of(Simplex s) { return static_cast<int>(s); }

  struct Candidate {
    Index i{};                // The snap point's row [0, np_); indexes its position and tolerance.
    Point3 aq;                // The projection of the point onto the mesh (closest point).
    double d2{};              // The squared distance from the point to the mesh.
    Index fi{};               // The projected face.
    std::array<double, 3> l;  // The barycentric coordinates of the projection.
    std::array<Simplex, 7> order;  // The seven simplices, nearest centroid first.
  };

  // A vertex on an edge, at parameter t from the edge's smaller-id endpoint.
  struct EdgeVertex {
    double t{};
    Index v{};
  };

  // A face's 2D frame: vertex 0 at the origin, edge 0->1 on x.
  struct Frame {
    Point3 origin;
    Vector3 e1;
    Vector3 e2;
  };

  // A face's mutable snapping state.
  struct Patch {
    std::pair<Point3, Point3> box;  // AABB currently registered in face_grid_
    std::vector<Index> interior;
    Faces faces;  // cached triangulation; empty until first computed
  };

 public:
  // A point snaps only if its distance to the mesh is <= max_distance and both it and its
  // projection lie in bbox. Vertices, points, and bbox are untransformed; the snapper applies aniso
  // (so an anisotropic resolution is respected), then emits untransformed positions. bbox stays
  // untransformed (rotating its AABB would inflate it), each point mapped back for the test.
  //
  // When bbox excludes the boundary by at least one face, the boundary is provably untouched:
  // snapping only moves a vertex or subdivides an edge of the face holding the (in-bbox)
  // projection, so the touched feature is interior. (In the pipeline bbox is first_extended_bbox.)
  Snapper(const Mesh& mesh, const Points3& points, const VecX& tolerances, double resolution,
          const geometry::Bbox3& bbox, double max_distance, const Mat3& aniso = Mat3::Identity())
      : nv_(mesh.vertices().rows()),
        np_(points.rows()),
        mesh_((mesh.faces().array() + np_).matrix()),  // shift original vertices to rows [np_, .)
        bbox_(bbox),
        aniso_inv_(aniso.inverse()),
        max_distance_(max_distance),
        snap_grid_(max_distance, np_),
        face_grid_(resolution, mesh.faces().rows()),
        patches_(mesh_.num_faces()) {
    if (tolerances.size() != 0 && tolerances.size() != points.rows()) {
      throw std::invalid_argument("tolerances must be empty or have one entry per point");
    }

    p_.resize(np_ + nv_, 3);
    ap_.resize(np_ + nv_, 3);
    aq_.resize(np_ + nv_, 3);
    p_.topRows(np_) = points;
    p_.bottomRows(nv_) = mesh.vertices();
    ap_.topRows(np_) = geometry::transform_points<3>(aniso, points);
    ap_.bottomRows(nv_) = geometry::transform_points<3>(aniso, mesh.vertices());
    aq_.bottomRows(nv_) = ap_.bottomRows(nv_);  // the anchor starts at the position; only ap_ moves

    VecX tols = tolerances;
    if (tols.size() == 0) {
      tols = VecX::Zero(np_);
    }
    snap_grid_.insert_balls(ap_.topRows(np_), tols);
    snap_tols2_ = tols.cwiseAbs2();

    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      auto f = mesh_.face(fi);
      auto ps = aq_(f, kAll);
      Point3 lo = ps.colwise().minCoeff();
      Point3 hi = ps.colwise().maxCoeff();
      face_grid_.insert(fi, lo, hi);
      patches_.at(fi).box = {lo, hi};
    }

    auto candidates = build_candidates();
    std::vector<bool> placed(candidates.size(), false);

    // Pass 1: vertex moves only (the leading vertex run of each cascade). Doing all moves before
    // any edge/face snap lets a vertex absorb its point before a snap subdivides the patch into
    // slivers.
    for (std::size_t i = 0; i < candidates.size(); i++) {
      const auto& cand = candidates.at(i);
      if (already_satisfied(cand)) {
        placed.at(i) = true;
        stats_.satisfied++;
        continue;
      }
      for (auto s : cand.order) {
        if (index_of(s) >= index_of(Simplex::kEdge12)) {
          break;  // an edge or the face is nearer than the remaining vertices: defer to pass 2
        }
        if (try_place(cand, s)) {
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
      // A vertex point the single-face cascade missed gets one more try across the vertex's
      // umbrella.
      if (!ok && index_of(cand.order.front()) <= index_of(Simplex::kVertex2)) {
        ok = try_vertex_umbrella(cand, mesh_.face(cand.fi)(index_of(cand.order.front())));
      }
      if (!ok) {
        stats_.dropped++;
      }
    }

    result_ = emit();
  }

  Mesh result() && { return std::move(result_); }

  const Stats& stats() const { return stats_; }

 private:
  // Whether the partially snapped mesh already passes within the point's tolerance, so snapping
  // would barely move it and only over-subdivide; checks the projected patch and the patches across
  // its edges.
  bool already_satisfied(const Candidate& cand) {
    auto patch_honors = [&](Index fi) {
      for (auto f : patch_faces(fi).rowwise()) {
        if (honored_by(cand.i, f)) {
          return true;
        }
      }
      return false;
    };
    if (patch_honors(cand.fi)) {
      return true;
    }
    auto cf = mesh_.face(cand.fi);
    for (auto k = 0; k < 3; k++) {
      Edge e{cf(k), cf((k + 1) % 3)};
      for (auto fj : mesh_.faces_of(e)) {
        if (fj != cand.fi && patch_honors(fj)) {
          return true;
        }
      }
    }
    return false;
  }

  // Project each point, rank its face's seven simplex centroids by distance to the projection
  // (a Voronoi classification), and sort the candidates by distance to the mesh.
  std::vector<Candidate> build_candidates() {
    const auto& V = aq_;

    std::vector<Candidate> candidates;
    candidates.reserve(np_);
    for (Index i = 0; i < np_; i++) {
      Point3 p = p_.row(i);
      Point3 ap = ap_.row(i);

      // The nearest face within max_distance; classification skips anything farther, so a
      // max_distance-radius query over face_grid_ suffices.
      int fi = -1;
      Point3 aq;
      auto d2 = std::numeric_limits<double>::infinity();
      Point3 lo = (ap.array() - max_distance_).matrix();
      Point3 hi = (ap.array() + max_distance_).matrix();
      face_grid_.for_each(lo, hi, [&](Index fj) {
        auto f = mesh_.face(fj);
        Point3 cp;
        auto dd = point_triangle_closest(ap, V.row(f(0)), V.row(f(1)), V.row(f(2)), cp);
        if (dd < d2) {
          d2 = dd;
          fi = static_cast<int>(fj);
          aq = cp;
        }
        return true;
      });

      if (fi < 0 || d2 > max_distance_ * max_distance_) {
        stats_.skipped++;
        continue;
      }
      Point3 q = geometry::transform_point<3>(aniso_inv_, aq);
      if (!bbox_.contains(p) || !bbox_.contains(q)) {
        stats_.skipped++;
        continue;
      }

      auto f = mesh_.face(fi);
      Point3 a = V.row(f(0));
      Point3 b = V.row(f(1));
      Point3 c = V.row(f(2));

      Vector3 l;
      igl::barycentric_coordinates(aq, a, b, c, l);

      // The centroid of each simplex, indexed by Simplex (vertices, edge midpoints, face).
      std::array<Point3, 7> sites{
          a, b, c, 0.5 * (b + c), 0.5 * (c + a), 0.5 * (a + b), (a + b + c) / 3.0};
      std::array<Simplex, 7> order{Simplex::kVertex0, Simplex::kVertex1, Simplex::kVertex2,
                                   Simplex::kEdge12,  Simplex::kEdge20,  Simplex::kEdge01,
                                   Simplex::kFace};
      std::ranges::sort(order, [&](Simplex s, Simplex t) {
        return (aq - sites.at(index_of(s))).squaredNorm() <
               (aq - sites.at(index_of(t))).squaredNorm();
      });

      candidates.push_back(
          {.i = i, .aq = aq, .d2 = d2, .fi = fi, .l = {l(0), l(1), l(2)}, .order = order});
    }

    // Least-distorting first (by distance to the mesh): each shared feature is claimed by the
    // candidate that moves it least. Ordering by tangential offset instead folds patches into
    // overhangs.
    std::ranges::sort(candidates, [this](const Candidate& x, const Candidate& y) {
      return std::make_tuple(x.d2, ap_(x.i, 0), ap_(x.i, 1), ap_(x.i, 2)) <
             std::make_tuple(y.d2, ap_(y.i, 0), ap_(y.i, 1), ap_(y.i, 2));
    });
    return candidates;
  }

  // Whether any changed triangle, valid over its flat on-surface positions, is collinear once
  // emitted at its moved 3D positions -- a degeneracy the flat triangulation cannot see.
  bool creates_degenerate(const boost::unordered_flat_map<Index, Faces>& changed) {
    for (const auto& [fi, faces] : changed) {
      auto scale = original_normal(fi).norm();  // twice the original face's area
      for (auto f : faces.rowwise()) {
        if (emitted_normal(f).norm() <= 1e-9 * scale) {
          return true;
        }
      }
    }
    return false;
  }

  double dist2(const Point3& p, const Face& f) const {
    return point_triangle_dist2(p, ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  Mesh emit() {
    std::vector<Face> faces;
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      for (auto f : patch_faces(fi).rowwise()) {
        faces.push_back(f);
      }
    }

    // Drop vertices no face references (chain thinning orphans the inserted ones it removes).
    auto n_all = np_ + nv_;
    std::vector<bool> used(n_all, false);
    for (const auto& f : faces) {
      for (auto v : f) {
        used.at(v) = true;
      }
    }
    // Number the original vertices (rows [np_, .)) first, then the inserted snap rows, so the
    // output keeps the input vertex order and an un-snapped run emits unchanged.
    std::vector<Index> vv(n_all, -1);
    Index n = 0;
    for (Index v = np_; v < n_all; v++) {
      if (used.at(v)) {
        vv.at(v) = n++;
      }
    }
    for (Index v = 0; v < np_; v++) {
      if (used.at(v)) {
        vv.at(v) = n++;
      }
    }

    Points3 vertices(n, 3);
    for (Index v = 0; v < n_all; v++) {
      if (used.at(v)) {
        vertices.row(vv.at(v)) = p_.row(v);  // exact untransformed position; no aniso round-trip
      }
    }

    Faces f(static_cast<Index>(faces.size()), 3);
    for (Index i = 0; i < f.rows(); i++) {
      f(i, 0) = vv.at(faces.at(i)(0));
      f(i, 1) = vv.at(faces.at(i)(1));
      f(i, 2) = vv.at(faces.at(i)(2));
    }
    return {std::move(vertices), std::move(f)};
  }

  Vector3 emitted_normal(const Face& f) const {
    return normal(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  // The 2D frame of the original (unsnapped) face fi.
  Frame frame(Index fi) const {
    auto f = mesh_.face(fi);
    Point3 a = aq_.row(f(0));
    Vector3 ab = aq_.row(f(1)) - a;
    Vector3 ac = aq_.row(f(2)) - a;
    Vector3 n = ab.cross(ac);
    if (!(n.squaredNorm() > 0.0)) {
      throw std::runtime_error(std::format("face {} is degenerate", fi));
    }
    return Frame{.origin = a, .e1 = ab.normalized(), .e2 = n.cross(ab).normalized()};
  }

  // Whether snap point i lies within its tolerance of face f.
  bool honored_by(Index i, const Face& f) const { return dist2(ap_.row(i), f) <= snap_tols2_(i); }

  // Whether point i is within its tolerance of the faces incident to v. Checking only those, not
  // the neighbouring faces too, can only over-reject a re-move, never dishonor.
  bool honored_by_surface(Index v, Index i) {
    for (auto fi : mesh_.incident(v)) {
      for (auto f : patch_faces(fi).rowwise()) {
        if ((f.array() == v).any() && honored_by(i, f)) {
          return true;
        }
      }
    }
    return false;
  }

  // The snap points the surface around v currently honors, found via the grid over v's patch AABB.
  std::vector<Index> honored_points_around(Index v) {
    Point3 lo = Point3::Constant(std::numeric_limits<double>::infinity());
    Point3 hi = -lo;
    for (auto fi : mesh_.incident(v)) {
      for (auto f : patch_faces(fi).rowwise()) {
        if ((f.array() == v).any()) {
          for (auto w : f) {
            lo = lo.cwiseMin(ap_.row(w));
            hi = hi.cwiseMax(ap_.row(w));
          }
        }
      }
    }
    std::vector<Index> honored;
    snap_grid_.for_each(lo, hi, [&](Index i) {
      if (honored_by_surface(v, i)) {
        honored.push_back(i);
      }
      return true;
    });
    return honored;
  }

  // Inserts the point into face fi's interior; reverts unless the re-triangulation uses it (it
  // projects inside the patch) and stays free of self-intersection.
  bool insert_in_face(const Candidate& cand, Index fi) {
    auto new_v = cand.i;       // the snap point's row; p_/ap_ already hold its position
    aq_.row(new_v) = cand.aq;  // the snap point's on-surface projection
    auto& interior = patches_.at(fi).interior;
    interior.push_back(new_v);

    bool simple = true;
    auto faces = triangulate_patch(fi, &simple);
    bool used = (faces.array() == new_v).any();
    boost::unordered_flat_map<Index, Faces> changed{{fi, faces}};
    auto revert = [&] { interior.pop_back(); };
    if (!simple || !used || creates_degenerate(changed) || over_pulled(changed) ||
        self_intersects(changed)) {
      revert();
      return false;
    }

    patches_.at(fi).faces = std::move(faces);
    reindex_patch(fi);
    stats_.inserted_in_faces++;
    return true;
  }

  // Inserts the point on edge e at parameter t from e's smaller-id endpoint, subdividing both its
  // incident patches; reverts unless the re-triangulations stay free of self-intersection.
  bool insert_on_edge(const Candidate& cand, const Edge& e, double t) {
    auto incident_faces = mesh_.faces_of(e);

    auto new_v = cand.i;  // the snap point's row; p_/ap_ already hold its position
    aq_.row(new_v) = (1.0 - t) * aq_.row(e.a) + t * aq_.row(e.b);  // on the original edge
    auto& chain = edge_chains_[e];
    chain.insert(std::ranges::lower_bound(chain, t, {}, &EdgeVertex::t), {.t = t, .v = new_v});

    boost::unordered_flat_map<Index, Faces> changed;
    bool simple = true;
    for (auto fi : incident_faces) {
      bool patch_simple = true;
      changed[fi] = triangulate_patch(fi, &patch_simple);
      simple = simple && patch_simple;
    }
    auto revert = [&] {
      std::erase_if(chain, [new_v](const EdgeVertex& x) { return x.v == new_v; });
      if (chain.empty()) {
        edge_chains_.erase(e);
      }
    };
    if (!simple || creates_degenerate(changed) || over_pulled(changed) ||
        self_intersects(changed)) {
      revert();
      return false;
    }

    for (auto fi : incident_faces) {
      patches_.at(fi).faces = changed.at(fi);
      reindex_patch(fi);
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
    auto f = mesh_.face(fi);
    return normal(aq_.row(f(0)), aq_.row(f(1)), aq_.row(f(2)));
  }

  // Whether any emitted triangle is folded over -- normal opposing its original's. An acute feature
  // up to vertical is kept; a fin there is under-resolution, not a fold.
  bool over_pulled(const boost::unordered_flat_map<Index, Faces>& changed) {
    for (const auto& [fi, faces] : changed) {
      auto n = original_normal(fi);
      for (auto f : faces.rowwise()) {
        if (emitted_normal(f).dot(n) < 0.0) {
          return true;
        }
      }
    }
    return false;
  }

  // The cached triangulation of a patch (computed on first use).
  const Faces& patch_faces(Index fi) {
    auto& patch = patches_.at(fi);
    if (patch.faces.rows() == 0) {
      patch.faces = triangulate_patch(fi);
    }
    return patch.faces;
  }

  // Drops p onto the face frame fr (its normal component removed).
  static Point2 project(const Frame& fr, const Point3& p) {
    Vector3 d = p - fr.origin;
    return Point2{d.dot(fr.e1), d.dot(fr.e2)};
  }

  // Refreshes fi's grid entry to the current bbox of its committed sub-faces. Call after a commit
  // that moved or re-triangulated the patch.
  void reindex_patch(Index fi) {
    auto& patch = patches_.at(fi);
    const auto& [lo0, hi0] = patch.box;
    face_grid_.remove(fi, lo0, hi0);
    auto inf = std::numeric_limits<double>::infinity();
    Point3 lo = Point3::Constant(inf);
    Point3 hi = Point3::Constant(-inf);
    for (auto f : patch_faces(fi).rowwise()) {
      auto ps = ap_(f, kAll);
      lo = lo.cwiseMin(Point3(ps.colwise().minCoeff()));
      hi = hi.cwiseMax(Point3(ps.colwise().maxCoeff()));
    }
    face_grid_.insert(fi, lo, hi);
    patch.box = {lo, hi};
  }

  // The snapper's one geometric acceptance test: the flat triangulation is always valid, so all
  // that remains is to forbid an actual self-intersection of the emitted mesh.
  bool self_intersects(const boost::unordered_flat_map<Index, Faces>& changed) {
    boost::unordered_flat_set<Index> changed_ids;
    std::vector<Face> changed_faces;
    for (const auto& [fi, faces] : changed) {
      changed_ids.insert(fi);
      for (auto f : faces.rowwise()) {
        changed_faces.push_back(f);
      }
    }
    // The predicate runs in the untransformed frame (p_), where defects are judged, matching the
    // defect finder; the broad-phase below stays in the aniso frame of mesh_.
    auto crosses = [&](const Face& a, const Face& b) {
      return triangles_intersect(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)), p_.row(b(0)),
                                 p_.row(b(1)), p_.row(b(2)), num_shared_vertices(a, b));
    };
    // Query per changed face against the committed patches near its exact AABB (face_grid_ tracks
    // their current geometry, so no margin is needed), rather than pooling all neighborhoods.
    boost::unordered_flat_set<Index> candidates;
    for (const auto& a : changed_faces) {
      for (const auto& b : changed_faces) {
        if (crosses(a, b)) {
          return true;
        }
      }
      auto aps = ap_(a, kAll);
      Point3 lo = aps.colwise().minCoeff();
      Point3 hi = aps.colwise().maxCoeff();
      candidates.clear();
      face_grid_.for_each(lo, hi, [&](Index fj) {
        if (!changed_ids.contains(fj)) {
          candidates.insert(fj);
        }
        return true;
      });
      for (auto fj : candidates) {
        for (auto b : patch_faces(fj).rowwise()) {
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
  Faces triangulate_patch(Index fi, bool* simple = nullptr) {
    if (simple != nullptr) {
      *simple = true;
    }
    auto f = mesh_.face(fi);
    std::array<Index, 3> vertices{f(0), f(1), f(2)};

    auto has_chain = [&](const Edge& e) { return edge_chains_.contains(e); };
    if (patches_.at(fi).interior.empty() && !has_chain({vertices[0], vertices[1]}) &&
        !has_chain({vertices[1], vertices[2]}) && !has_chain({vertices[2], vertices[0]})) {
      Faces single(1, 3);
      single.row(0) = Face(vertices[0], vertices[1], vertices[2]);
      return single;
    }

    auto fr = frame(fi);
    std::vector<Point2> boundary;
    std::vector<Index> boundary_ids;
    // The original edge(s) each boundary vertex lies on, so the triangulation never cuts a diagonal
    // along a subdivided edge (see Triangulation). Vertex k lies on patch edges k and (k + 2) % 3.
    std::vector<std::array<int, 2>> boundary_edges;
    auto add_vertex = [&](int k) {
      boundary_ids.push_back(vertices.at(k));
      // Project the *flat* (on-surface) position, not the snapped target, so the 2D polygon is
      // always valid and consistently wound; folds of the moved mesh are caught downstream.
      boundary.push_back(project(fr, aq_.row(vertices.at(k))));
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
        boundary.push_back(project(fr, aq_.row(v)));
        boundary_edges.push_back({edge, -1});
      };
      if (from < to) {
        for (const auto& x : chain) {
          append(x.v);
        }
      } else {
        for (auto i = static_cast<Index>(chain.size()) - 1; i >= 0; i--) {
          append(chain.at(i).v);
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
    for (auto v : patches_.at(fi).interior) {
      interior_ids.push_back(v);
      interior.push_back(project(fr, aq_.row(v)));
    }

    auto nb = static_cast<Index>(boundary_ids.size());
    Triangulation triangulation(boundary, interior, std::move(boundary_edges));
    if (simple != nullptr) {
      *simple = triangulation.simple();
    }
    auto map = [&](Index i) { return i < nb ? boundary_ids.at(i) : interior_ids.at(i - nb); };
    const auto& tf = triangulation.faces();
    Faces faces(tf.rows(), 3);
    for (Index r = 0; r < tf.rows(); r++) {
      faces.row(r) = Face(map(tf(r, 0)), map(tf(r, 1)), map(tf(r, 2)));
    }
    return faces;
  }

  // Subdivides both patches incident to the edge. `i` is the local index of the vertex opposite
  // the edge; j and k are the edge's endpoints.
  bool try_edge(const Candidate& cand, int i) {
    auto j = (i + 1) % 3;
    auto k = (i + 2) % 3;
    if (!(cand.l.at(j) + cand.l.at(k) > 0.0)) {
      return false;
    }
    auto f = mesh_.face(cand.fi);
    auto vj = f(j);
    auto vk = f(k);
    auto t = cand.l.at(k) / (cand.l.at(j) + cand.l.at(k));
    if (vj > vk) {
      std::swap(vj, vk);
      t = 1.0 - t;
    }
    return insert_on_edge(cand, {vj, vk}, t);
  }

  bool try_interior(const Candidate& cand) { return insert_in_face(cand, cand.fi); }

  bool try_place(const Candidate& cand, Simplex s) {
    switch (s) {
      case Simplex::kVertex0:
      case Simplex::kVertex1:
      case Simplex::kVertex2:
        return try_vertex(cand, mesh_.face(cand.fi)(index_of(s)));
      case Simplex::kEdge12:
      case Simplex::kEdge20:
      case Simplex::kEdge01:
        return try_edge(cand, index_of(s) - index_of(Simplex::kEdge12));
      case Simplex::kFace:
        return try_interior(cand);
    }
    return false;  // unreachable; all simplices are handled above
  }

  // Moves v onto the candidate's point, but only if every point the surface around v currently
  // honors stays honored -- so no move (a first move or a re-move to a farther point) dishonors an
  // already-served point.
  bool try_vertex(const Candidate& cand, Index v) {
    auto honored = honored_points_around(v);

    Point3 p = p_.row(v);  // for revert
    Point3 ap = ap_.row(v);
    p_.row(v) = p_.row(cand.i);  // tentative
    ap_.row(v) = ap_.row(cand.i);
    boost::unordered_flat_map<Index, Faces> changed;
    for (auto fi : mesh_.incident(v)) {
      changed[fi] = patch_faces(fi);
    }
    bool dishonors =
        std::ranges::any_of(honored, [&](Index i) { return !honored_by_surface(v, i); });
    if (creates_degenerate(changed) || over_pulled(changed) || self_intersects(changed) ||
        dishonors) {
      p_.row(v) = p;
      ap_.row(v) = ap;
      return false;
    }

    for (auto fi : mesh_.incident(v)) {
      reindex_patch(fi);
    }
    stats_.moved_vertices++;
    return true;
  }

  // Retry a vertex-classified point across v's whole umbrella, nearest first: the cascade only saw
  // the single AABB face, so a spoke or face of another incident face may still take it.
  bool try_vertex_umbrella(const Candidate& cand, Index v) {
    const auto& V = aq_;

    // The edges of the cascade's own face that contain v were already tried; skip them.
    boost::unordered_flat_set<Edge, EdgeHash> queued;
    auto cf = mesh_.face(cand.fi);
    for (auto k = 0; k < 3; k++) {
      Index a = cf(k);
      Index b = cf((k + 1) % 3);
      if (a == v || b == v) {
        queued.insert({a, b});
      }
    }

    struct Op {
      double d2{};
      Index a{-1};
      Index b{-1};
      Index fi{-1};
      double t{};
    };
    std::vector<Op> ops;
    for (auto fi : mesh_.incident(v)) {
      if (fi == cand.fi) {
        continue;
      }
      auto f = mesh_.face(fi);
      Point3 a = V.row(f(0));
      Point3 b = V.row(f(1));
      Point3 c = V.row(f(2));
      Point3 centroid = a + ((b - a) + (c - a)) / 3.0;
      ops.push_back({.d2 = (cand.aq - centroid).squaredNorm(), .fi = fi});
      for (auto k = 0; k < 3; k++) {
        Index x = f(k);
        Index y = f((k + 1) % 3);
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
        auto t = (cand.aq - va).dot(d) / d.squaredNorm();
        if (!(t > 0.0 && t < 1.0)) {
          continue;  // projects onto an endpoint
        }
        Point3 mid = va + 0.5 * d;
        ops.push_back({.d2 = (cand.aq - mid).squaredNorm(), .a = e.a, .b = e.b, .t = t});
      }
    }

    std::ranges::sort(ops, [](const Op& x, const Op& y) {
      return std::make_tuple(x.d2, x.fi, x.a, x.b) < std::make_tuple(y.d2, y.fi, y.a, y.b);
    });
    for (const auto& op : ops) {
      auto ok = op.fi >= 0 ? insert_in_face(cand, op.fi) : insert_on_edge(cand, {op.a, op.b}, op.t);
      if (ok) {
        return true;
      }
    }
    return false;
  }

  const Index nv_;
  const Index np_;
  AbstractMesh mesh_;
  geometry::Bbox3 bbox_;
  Mat3 aniso_inv_;
  double max_distance_;
  VecX snap_tols2_;
  SpatialGrid snap_grid_;  // snap-point broad-phase for the re-move honor check
  SpatialGrid face_grid_;  // committed-patch broad-phase for the self-intersection guard
  std::vector<Patch> patches_;
  // Positions indexed by vertex row: row i (< np_) is snap point i; row np_ + v is original vertex
  // v. mesh_ is built from faces shifted by np_, so mesh_.face already yields these rows.
  Points3 p_;   // untransformed position emit() outputs (snap points pass through exactly)
  Points3 ap_;  // aniso position, for the geometry/self-intersection tests; a moving vertex's
                // original row is overwritten, snap-point rows are immutable
  Points3 aq_;  // aniso on-surface anchor the triangulation projects; original rows never move (the
                // immutable original geometry), snap rows hold each insert's projection
  boost::unordered_flat_map<Edge, std::vector<EdgeVertex>, EdgeHash> edge_chains_;
  Stats stats_;
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
