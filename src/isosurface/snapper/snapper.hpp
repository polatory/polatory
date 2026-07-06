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
// already snapped stay honored, letting it migrate out to a contour tip (see try_vertex).
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
    Index fi{};               // The projected face.
    Point3 aq;                // The projection of the point onto the mesh (closest point).
    double d2{};              // The squared distance from the point to the mesh.
    std::array<double, 3> l;  // The barycentric coordinates of the projection.
    std::array<Simplex, 7> order;  // The seven simplices, nearest centroid first.
  };

  // A vertex on an edge, at parameter t from the edge's smaller-id endpoint.
  struct EdgeVertex {
    double t{};
    Index v{};
  };

  // A face's 2D frame: origin at vertex 0, e1 the unit edge 0->1, e2 the in-plane perpendicular.
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
    std::vector<Index> honored;
    bool honored_valid = false;
  };

 public:
  // A point snaps only if its distance to the mesh is <= max_distance. Vertices and points are
  // untransformed; the snapper applies aniso (so an anisotropic resolution is respected), then
  // emits untransformed positions.
  Snapper(const Mesh& mesh, const Points3& points, const VecX& tolerances, double resolution,
          const Mat3& aniso, double max_distance)
      : nv_(mesh.vertices().rows()),
        np_(points.rows()),
        mesh_((mesh.faces().array() + np_).matrix()),  // shift original vertices to rows [np_, .)
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
      auto aqs = aq_(f, kAll);
      Point3 lo = aqs.colwise().minCoeff();
      Point3 hi = aqs.colwise().maxCoeff();
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
      for (auto s : cand.order) {
        if (try_place(cand, s)) {
          ok = true;
          break;
        }
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
    if (honored_by_patch(cand.i, cand.fi)) {
      return true;
    }
    for (auto k = 0; k < 3; k++) {
      auto h = mesh_.halfedge(cand.fi, k);
      Index fj = mesh_.face(mesh_.opposite(h));
      if (fj >= 0 && honored_by_patch(cand.i, fj)) {
        return true;
      }
    }
    return false;
  }

  // The k-th rollback slot, snapshotted into by copy-assignment so its storage is reused.
  Patch& backup_slot(std::size_t k) {
    if (k == backups_.size()) {
      backups_.emplace_back();
    }
    return backups_.at(k);
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
      Index best_fi = -1;
      Point3 best_aq;
      auto best_d2 = std::numeric_limits<double>::infinity();

      Point3 lo = (ap.array() - max_distance_).matrix();
      Point3 hi = (ap.array() + max_distance_).matrix();
      face_grid_.for_each(lo, hi, [&](Index fi) {
        const auto& [blo, bhi] = patches_.at(fi).box;
        Point3 aq = ap.cwiseMax(blo).cwiseMin(bhi);
        if ((ap - aq).squaredNorm() < best_d2) {
          auto f = mesh_.face(fi);
          auto d2 = point_triangle_closest(ap, V.row(f(0)), V.row(f(1)), V.row(f(2)), aq);
          if (d2 < best_d2) {
            best_fi = fi;
            best_aq = aq;
            best_d2 = d2;
          }
        }
        return true;
      });

      if (best_fi < 0 || best_d2 > max_distance_ * max_distance_) {
        stats_.skipped++;
        continue;
      }

      auto f = mesh_.face(best_fi);
      Point3 a = V.row(f(0));
      Point3 b = V.row(f(1));
      Point3 c = V.row(f(2));

      Vector3 l;
      igl::barycentric_coordinates(best_aq, a, b, c, l);

      // The centroid of each simplex, indexed by Simplex (vertices, edge midpoints, face).
      std::array<Point3, 7> sites{
          a, b, c, 0.5 * (b + c), 0.5 * (c + a), 0.5 * (a + b), (a + b + c) / 3.0};
      std::array<Simplex, 7> order{Simplex::kVertex0, Simplex::kVertex1, Simplex::kVertex2,
                                   Simplex::kEdge12,  Simplex::kEdge20,  Simplex::kEdge01,
                                   Simplex::kFace};
      std::ranges::sort(order, [&](auto s, auto t) {
        return (best_aq - sites.at(index_of(s))).squaredNorm() <
               (best_aq - sites.at(index_of(t))).squaredNorm();
      });

      candidates.push_back({.i = i,
                            .fi = best_fi,
                            .aq = best_aq,
                            .d2 = best_d2,
                            .l = {l(0), l(1), l(2)},
                            .order = order});
    }

    // Least-distorting first (by distance to the mesh): each shared feature is claimed by the
    // candidate that moves it least. Ordering by tangential offset instead folds patches into
    // overhangs.
    std::ranges::sort(candidates, [this](const auto& x, const auto& y) {
      return std::make_tuple(x.d2, ap_(x.i, 0), ap_(x.i, 1), ap_(x.i, 2)) <
             std::make_tuple(y.d2, ap_(y.i, 0), ap_(y.i, 1), ap_(y.i, 2));
    });
    return candidates;
  }

  // Whether any emitted triangle (at its moved 3D positions) is degenerate -- collinear, a
  // zero-area sliver the flat triangulation cannot see -- or folded over, its normal opposing the
  // original face's. An acute feature up to vertical is kept; a fin there is under-resolution, not
  // a fold.
  bool degenerate_or_folded(const boost::unordered_flat_map<Index, Faces>& changed) {
    auto normal = [](const Point3& a, const Point3& b, const Point3& c) {
      return Vector3((b - a).cross(c - a));
    };
    for (const auto& [fi, faces] : changed) {
      auto f = mesh_.face(fi);
      auto n = normal(aq_.row(f(0)), aq_.row(f(1)), aq_.row(f(2)));
      auto scale = n.norm();  // twice the original face's area
      for (auto nf : faces.rowwise()) {
        auto nn = normal(ap_.row(nf(0)), ap_.row(nf(1)), ap_.row(nf(2)));
        if (nn.norm() <= 1e-9 * scale || nn.dot(n) < 0.0) {
          return true;
        }
      }
    }
    return false;
  }

  Mesh emit() {
    Points3 vertices(p_.rows(), 3);
    Faces faces(mesh_.num_faces() + 2 * (stats_.inserted_on_edges + stats_.inserted_in_faces), 3);
    Index nf = 0;
    Index nv = 0;
    std::vector<Index> vv(p_.rows(), -1);
    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      for (auto f : patch_faces(fi).rowwise()) {
        for (auto k = 0; k < 3; k++) {
          auto v = f(k);
          if (vv.at(v) < 0) {
            vv.at(v) = nv;
            vertices.row(nv) = p_.row(v);
            nv++;
          }
          faces(nf, k) = vv.at(v);
        }
        nf++;
      }
    }
    vertices.conservativeResize(nv, Eigen::NoChange);
    return {std::move(vertices), std::move(faces)};
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
  bool honored_by(Index i, const Face& f) const {
    auto aps = ap_(f, kAll);
    Point3 ap = ap_.row(i);
    Point3 lo = aps.colwise().minCoeff();
    Point3 hi = aps.colwise().maxCoeff();
    Point3 aq = ap.cwiseMax(lo).cwiseMin(hi);
    if ((ap - aq).squaredNorm() > snap_tols2_(i)) {
      return false;
    }
    return point_triangle_dist2(ap, aps.row(0), aps.row(1), aps.row(2)) <= snap_tols2_(i);
  }

  // Whether the committed mesh honors point i; call after a commit (reads the face grid).
  bool honored_by_mesh(Index i) {
    double tol = std::sqrt(snap_tols2_(i));
    Point3 ap = ap_.row(i);
    Point3 lo = (ap.array() - tol).matrix();
    Point3 hi = (ap.array() + tol).matrix();
    bool honored = false;
    face_grid_.for_each(lo, hi, [&](Index fi) {
      if (honored_by_patch(i, fi)) {
        honored = true;
        return false;
      }
      return true;
    });
    return honored;
  }

  // Whether patch fi's current triangulation honors point i.
  bool honored_by_patch(Index i, Index fi) {
    for (auto f : patch_faces(fi).rowwise()) {
      if (honored_by(i, f)) {
        return true;
      }
    }
    return false;
  }

  // Whether point i is within its tolerance of v's star (the faces incident to v).
  bool honored_by_star(Index i, Index v) {
    for (auto fi : mesh_.vertex_faces(v)) {
      for (auto f : patch_faces(fi).rowwise()) {
        if ((f.array() == v).any() && honored_by(i, f)) {
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

  // The snap points the given patches honor; each patch's set is cached until it is reindexed.
  std::vector<Index> points_honored_by_patches(
      const boost::container::static_vector<Index, 2>& patches) {
    std::vector<Index> honored;
    for (auto fi : patches) {
      auto& patch = patches_.at(fi);
      if (!patch.honored_valid) {
        patch.honored.clear();
        if (!snap_grid_.empty()) {
          const auto& [lo, hi] = patch.box;
          snap_grid_.for_each(lo, hi, [&](Index i) {
            if (honored_by_patch(i, fi)) {
              patch.honored.push_back(i);
            }
            return true;
          });
        }
        patch.honored_valid = true;
      }
      honored.insert(honored.end(), patch.honored.begin(), patch.honored.end());
    }
    return honored;
  }

  // The snap points the surface around v currently honors, found via the grid over v's patch AABB.
  std::vector<Index> points_honored_by_star(Index v) {
    Point3 lo = Point3::Constant(std::numeric_limits<double>::infinity());
    Point3 hi = -lo;
    for (auto fi : mesh_.vertex_faces(v)) {
      for (auto f : patch_faces(fi).rowwise()) {
        if ((f.array() == v).any()) {
          auto aps = ap_(f, kAll);
          lo = lo.cwiseMin(aps.colwise().minCoeff());
          hi = hi.cwiseMax(aps.colwise().maxCoeff());
        }
      }
    }
    std::vector<Index> honored;
    snap_grid_.for_each(lo, hi, [&](Index i) {
      if (honored_by_star(i, v)) {
        honored.push_back(i);
      }
      return true;
    });
    return honored;
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
      auto aps = ap_(f, kAll);
      lo = lo.cwiseMin(aps.colwise().minCoeff());
      hi = hi.cwiseMax(aps.colwise().maxCoeff());
    }
    face_grid_.insert(fi, lo, hi);
    patch.box = {lo, hi};
    patch.honored_valid = false;
  }

  // Rolls fi back to a pre-commit snapshot, re-registering it under its restored box. Recovers the
  // prior faces and honored cache without re-triangulating or recomputing them.
  void restore_patch(Index fi, Patch& backup) {
    const auto& [lo0, hi0] = patches_.at(fi).box;
    face_grid_.remove(fi, lo0, hi0);
    std::swap(patches_.at(fi), backup);
    const auto& [lo, hi] = patches_.at(fi).box;
    face_grid_.insert(fi, lo, hi);
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
    // Any two changed faces crossing each other.
    for (std::size_t i = 0; i + 1 < changed_faces.size(); i++) {
      for (std::size_t j = i + 1; j < changed_faces.size(); j++) {
        if (crosses(changed_faces.at(i), changed_faces.at(j))) {
          return true;
        }
      }
    }
    // Each changed face against the committed patches near its exact AABB (face_grid_ tracks their
    // current geometry, so no margin is needed), rather than pooling all neighborhoods.
    for (const auto& a : changed_faces) {
      auto aps = ap_(a, kAll);
      Point3 lo = aps.colwise().minCoeff();
      Point3 hi = aps.colwise().maxCoeff();
      bool hit = false;
      face_grid_.for_each(lo, hi, [&](Index fj) {
        if (changed_ids.contains(fj)) {
          return true;  // skip a changed face
        }
        for (auto b : patch_faces(fj).rowwise()) {
          if (crosses(a, b)) {
            hit = true;
            return false;
          }
        }
        return true;
      });
      if (hit) {
        return true;
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
    const auto& patch = patches_.at(fi);

    auto has_chain = [&](const Edge& e) { return edge_chains_.contains(e); };
    if (patch.interior.empty() && !has_chain({f(0), f(1)}) && !has_chain({f(1), f(2)}) &&
        !has_chain({f(2), f(0)})) {
      Faces single(1, 3);
      single.row(0) = f;
      return single;
    }

    auto fr = frame(fi);
    std::vector<Point2> boundary;
    std::vector<Index> boundary_ids;
    // The original edge(s) each boundary vertex lies on, so the triangulation never cuts a diagonal
    // along a subdivided edge (see Triangulation). Vertex k lies on patch edges k and (k + 2) % 3.
    std::vector<std::array<int, 2>> boundary_edges;
    auto add_vertex = [&](int k) {
      boundary_ids.push_back(f(k));
      // Project the *flat* (on-surface) position, not the snapped target, so the 2D polygon is
      // always valid and consistently wound; folds of the moved mesh are caught downstream.
      boundary.push_back(project(fr, aq_.row(f(k))));
      boundary_edges.push_back({k, (k + 2) % 3});
    };
    auto add_chain = [&](int edge) {
      auto from = f(edge);
      auto to = f((edge + 1) % 3);
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
    for (auto v : patch.interior) {
      interior.push_back(project(fr, aq_.row(v)));
    }

    auto nb = static_cast<Index>(boundary_ids.size());
    Triangulation triangulation(boundary, interior, std::move(boundary_edges));
    if (simple != nullptr) {
      *simple = triangulation.simple();
    }
    auto map = [&](Index i) { return i < nb ? boundary_ids.at(i) : patch.interior.at(i - nb); };
    const auto& tf = triangulation.faces();
    Faces faces(tf.rows(), 3);
    for (Index r = 0; r < tf.rows(); r++) {
      faces.row(r) = Face(map(tf(r, 0)), map(tf(r, 1)), map(tf(r, 2)));
    }
    return faces;
  }

  // Tries to insert the point on edge i (the local index of the vertex opposite it).
  bool try_edge(const Candidate& cand, int i) {
    auto j = (i + 1) % 3;
    auto k = (i + 2) % 3;
    if (!(cand.l.at(j) + cand.l.at(k) > 0.0)) {
      return false;
    }
    auto f = mesh_.face(cand.fi);
    Edge e{f(j), f(k)};
    if (e.a != f(j)) {
      std::swap(j, k);
    }
    auto t = cand.l.at(k) / (cand.l.at(j) + cand.l.at(k));

    auto incident_faces = mesh_.faces_of(e);
    if (incident_faces.size() < 2) {
      // A boundary edge is not a snap-insert target: it lies on the padding skirt (beyond the user
      // bbox, clipped off at the end), and splitting its lone triangle would move a boundary the
      // output discards. The cascade falls through to a face snap instead.
      return false;
    }

    auto honored = points_honored_by_patches(incident_faces);  // honored before the insert

    auto new_v = cand.i;  // the snap point's row; p_/ap_ already hold its position
    aq_.row(new_v) = aq_.row(e.a) + t * (aq_.row(e.b) - aq_.row(e.a));  // on the original edge
    auto& chain = edge_chains_[e];
    chain.insert(std::ranges::lower_bound(chain, t, {}, &EdgeVertex::t), {.t = t, .v = new_v});
    auto revert = [&] {
      std::erase_if(chain, [new_v](const EdgeVertex& x) { return x.v == new_v; });
      if (chain.empty()) {
        edge_chains_.erase(e);
      }
    };

    boost::unordered_flat_map<Index, Faces> changed;
    auto simple = true;
    for (auto fi : incident_faces) {
      changed[fi] = triangulate_patch(fi, &simple);
      if (!simple) {
        break;
      }
    }

    if (!simple || degenerate_or_folded(changed) || self_intersects(changed)) {
      revert();
      return false;
    }

    std::size_t b = 0;
    for (auto fi : incident_faces) {
      backup_slot(b++) = patches_.at(fi);
    }
    for (auto fi : incident_faces) {
      patches_.at(fi).faces = changed.at(fi);
      reindex_patch(fi);
    }

    if (!std::ranges::all_of(honored, [&](Index i) { return honored_by_mesh(i); })) {
      revert();
      b = 0;
      for (auto fi : incident_faces) {
        restore_patch(fi, backups_.at(b++));
      }
      return false;
    }

    stats_.inserted_on_edges++;
    return true;
  }

  // Tries to insert the point into its projected face's interior.
  bool try_face(const Candidate& cand) {
    auto fi = cand.fi;
    auto honored = points_honored_by_patches({fi});  // honored before the insert

    auto new_v = cand.i;       // the snap point's row; p_/ap_ already hold its position
    aq_.row(new_v) = cand.aq;  // the snap point's on-surface projection
    auto& interior = patches_.at(fi).interior;
    interior.push_back(new_v);
    auto revert = [&] { interior.pop_back(); };

    auto simple = true;
    auto faces = triangulate_patch(fi, &simple);
    bool used = (faces.array() == new_v).any();  // false if the point did not project inside
    boost::unordered_flat_map<Index, Faces> changed{{fi, faces}};

    if (!simple || !used || degenerate_or_folded(changed) || self_intersects(changed)) {
      revert();
      return false;
    }

    backup_slot(0) =
        patches_.at(fi);  // snapshot before the overwrite (still holds the added point)
    patches_.at(fi).faces = std::move(faces);
    reindex_patch(fi);

    if (!std::ranges::all_of(honored, [&](Index i) { return honored_by_mesh(i); })) {
      restore_patch(fi, backups_.at(0));  // prior faces and honored cache, no re-triangulation
      revert();                           // drop the point the snapshot carried
      return false;
    }

    stats_.inserted_in_faces++;
    return true;
  }

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
        return try_face(cand);
    }
    return false;  // unreachable; all simplices are handled above
  }

  // Tries to move v onto the candidate's point.
  bool try_vertex(const Candidate& cand, Index v) {
    auto honored = points_honored_by_star(v);  // honored before the move

    Point3 p = p_.row(v);  // for revert
    Point3 ap = ap_.row(v);
    p_.row(v) = p_.row(cand.i);  // tentative
    ap_.row(v) = ap_.row(cand.i);
    auto revert = [&] {
      p_.row(v) = p;
      ap_.row(v) = ap;
    };

    boost::unordered_flat_map<Index, Faces> changed;
    for (auto fi : mesh_.vertex_faces(v)) {
      changed[fi] = patch_faces(fi);
    }
    if (degenerate_or_folded(changed) || self_intersects(changed)) {
      revert();
      return false;
    }

    std::size_t b = 0;
    for (auto fi : mesh_.vertex_faces(v)) {
      backup_slot(b++) = patches_.at(fi);
    }
    for (auto fi : mesh_.vertex_faces(v)) {
      reindex_patch(fi);
    }

    if (!std::ranges::all_of(honored, [&](Index i) { return honored_by_mesh(i); })) {
      revert();
      b = 0;
      for (auto fi : mesh_.vertex_faces(v)) {
        restore_patch(fi, backups_.at(b++));
      }
      return false;
    }

    stats_.moved_vertices++;
    return true;
  }

  const Index nv_;
  const Index np_;
  AbstractMesh mesh_;
  Mat3 aniso_inv_;
  double max_distance_;
  VecX snap_tols2_;
  SpatialGrid snap_grid_;  // snap-point broad-phase for the re-move honor check
  SpatialGrid face_grid_;  // committed-patch broad-phase for the self-intersection guard
  std::vector<Patch> patches_;
  std::vector<Patch> backups_;  // reused scratch holding pre-commit patch state for rollback
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
