#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <boost/unordered/unordered_flat_set.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <queue>
#include <utility>
#include <vector>

#include "../abstract_mesh.hpp"
#include "../face_grid.hpp"
#include "../spatial_grid.hpp"
#include "../utility.hpp"

namespace polatory::isosurface::snapper {

using geometry::Points3;

// Edge-length optimization toward the range (kMinEdgeRatio, kMaxEdgeRatio) * res, scheduling splits
// and collapses through a priority queue (see schedule()): a long edge is bisected, a short one
// collapsed, most out-of-range first, re-queuing the edges each touches until all are in range or a
// guard defers them. The collapse's edge-length cap keeps it from feeding the split a long edge
// whose apex-connecting bisection re-adds a short one -- an otherwise unbounded split/collapse
// churn. Both steps are guarded against exposing a self-intersection MeshDefectsFinder would count:
// a midpoint split is geometry-preserving, but by dropping the shared-vertex count between
// overlapping faces it can turn a tolerated edge-adjacent fold into a counted one, so a split that
// would is deferred. A collapse is kept only if the dropped point stays within tolerance and the
// mesh stays manifold, unflipped, and self-intersection-free -- so the honor guard alone keeps a
// snapped feature point from being dropped when that would dishonor it. Geometry is in the
// aniso-transformed frame; the output is untransformed.
class EdgeOptimizer {
  using Point3 = geometry::Point3;
  using Vector3 = geometry::Vector3;

  // edges longer than this * res are split; a collapse may not create one this long either
  static constexpr double kMaxEdgeRatio = 4.0 / 3.0;
  // an edge shorter than this * res is a sliver, collapsed even if its vertex is not snapped
  static constexpr double kMinEdgeRatio = 3.0 / 4.0;

 public:
  EdgeOptimizer(const Mesh& mesh, const Points3& points, const VecX& tolerances, double resolution,
                const Mat3& aniso)
      : p_(mesh.vertices()),
        ap_(geometry::transform_points<3>(aniso, mesh.vertices())),
        mesh_(mesh.faces()),
        a_points_(geometry::transform_points<3>(aniso, points)),
        snap_grid_(resolution, points.rows()),
        face_grid_(resolution, mesh.faces().rows()),
        max_edge2_(kMaxEdgeRatio * resolution * (kMaxEdgeRatio * resolution)),
        min_edge2_(kMinEdgeRatio * resolution * (kMinEdgeRatio * resolution)) {
    VecX tols = tolerances;
    if (tols.size() == 0) {
      tols = VecX::Zero(a_points_.rows());
    }
    snap_grid_.insert_balls(a_points_, tols);
    snap_tols2_ = tols.cwiseAbs2();

    for (Index fi = 0; fi < mesh_.num_faces(); fi++) {
      index_face(fi);
    }
    schedule();
    result_ = emit();
  }

  Mesh result() && { return std::move(result_); }

 private:
  // A queued edge keyed by how far it is out of range; the queue takes the most out-of-range first
  // and re-checks each on pop, since a nearby split or collapse may have staled it.
  struct Item {
    Edge e;
    double urgency;
    bool operator<(const Item& other) const { return urgency < other.urgency; }
  };

  // Interleaves splitting long edges and collapsing short ones through a priority queue, most
  // out-of-range first (like the smoother's flip queue). A split enqueues the edges it creates and
  // a collapse those around the kept vertex, so a collapse's over-long edge is split and a split's
  // short edge is collapsed until every edge is in (kMinEdgeRatio, kMaxEdgeRatio) * res or an
  // operation is refused by a guard. The pipeline's relaxation regularizes the vertices between
  // calls, freeing edges a guard here had to defer (e.g. a sliver wedged between two long edges).
  void schedule() {
    std::priority_queue<Item> pq;
    auto enqueue = [&](const Edge& e) {
      if (!mesh_.has_edge(e)) {
        return;
      }
      auto len2 = (ap_.row(e.a) - ap_.row(e.b)).squaredNorm();
      double urgency = 0.0;
      if (len2 > max_edge2_) {
        urgency = len2 - max_edge2_;
      } else if (len2 < min_edge2_) {
        urgency = min_edge2_ - len2;
      }
      if (urgency > 0.0) {
        pq.push({e, urgency});
      }
    };
    mesh_.for_each_halfedge([&](Halfedge h) {
      if (mesh_.from(h) < mesh_.to(h)) {
        enqueue(Edge{mesh_.from(h), mesh_.to(h)});
      }
    });

    std::int64_t ops = 0;
    auto cap = 50 * std::max<std::int64_t>(mesh_.num_faces(), 1);  // backstop against a float cycle
    while (!pq.empty()) {
      auto e = pq.top().e;
      pq.pop();
      if (!mesh_.has_edge(e)) {
        continue;  // a prior operation removed it
      }
      auto len2 = (ap_.row(e.a) - ap_.row(e.b)).squaredNorm();
      bool changed = false;
      if (len2 > max_edge2_) {
        changed = split_edge(e, enqueue);
      } else if (len2 < min_edge2_) {
        changed = collapse_edge(e, enqueue);
      }
      if (changed && ++ops > cap) {
        break;
      }
    }
  }

  // Bisects interior edge e at its midpoint, guarded (like the batch split) against exposing a
  // counted self-intersection; enqueues the four edges it creates. Returns whether it split.
  template <class Enqueue>
  bool split_edge(const Edge& e, const Enqueue& enqueue) {
    auto h0 = mesh_.halfedge_of(e.a, e.b);
    auto opp = mesh_.opposite(h0);
    if (!opp.is_valid()) {
      return false;  // a boundary edge lies on the clipped-off skirt; leave it
    }
    auto fi0 = mesh_.face(h0);
    auto fi1 = mesh_.face(opp);
    auto c = mesh_.apex(h0);
    auto d = mesh_.apex(opp);
    Index nv = p_.rows();
    Point3 m = 0.5 * (p_.row(e.a) + p_.row(e.b));
    if (split_reveals_intersection(mesh_, p_, e, c, d, nv, m, fi0, fi1)) {
      return false;
    }

    p_.conservativeResize(nv + 1, Eigen::NoChange);
    ap_.conservativeResize(nv + 1, Eigen::NoChange);
    p_.row(nv) = m;
    ap_.row(nv) =
        0.5 * (ap_.row(e.a) + ap_.row(e.b));  // aniso is linear: midpoint maps to midpoint
    unindex_face(fi0);
    unindex_face(fi1);
    auto first_new = mesh_.num_faces();
    mesh_.insert_on_edge(e, nv);
    index_face(fi0);
    index_face(fi1);
    for (auto fi = first_new; fi < mesh_.num_faces(); fi++) {
      index_face(fi);
    }
    enqueue(Edge{e.a, nv});
    enqueue(Edge{nv, e.b});
    enqueue(Edge{nv, c});
    enqueue(Edge{nv, d});
    return true;
  }

  // Collapses short edge e by dropping one endpoint onto the other (try_collapse picks the
  // least-distorting admissible direction, so a feature vertex is kept); enqueues the kept vertex's
  // edges. Returns whether it collapsed.
  template <class Enqueue>
  bool collapse_edge(const Edge& e, const Enqueue& enqueue) {
    Index kept = try_collapse(e.a);
    if (kept < 0) {
      kept = try_collapse(e.b);
    }
    if (kept < 0) {
      return false;
    }
    for (auto h : mesh_.vertex_outgoing_halfedges(kept)) {
      enqueue(Edge{kept, mesh_.to(h)});
    }
    return true;
  }

  // MeshDefectsFinder's exact verdict for a face pair sharing vertex vi, given a position lookup:
  // skip the pair if they share a second vertex (edge-adjacent, which the finder ignores),
  // otherwise report an intersection if either face's edge opposite vi pierces the other face.
  // Matching this exactly -- rather than the edge-adjacent-skipping triangles_intersect -- is what
  // lets a local guard promise the finder counts zero.
  template <class Pos>
  static bool finder_pair_hits(Index vi, const Face& f, const Face& g, const Pos& pos) {
    auto opposite_edge = [](const Face& t, Index v, Index& o0, Index& o1) {
      o0 = -1;
      o1 = -1;
      auto n = 0;
      for (auto k = 0; k < 3; k++) {
        if (t(k) == v) {
          n++;
        } else if (o0 < 0) {
          o0 = t(k);
        } else {
          o1 = t(k);
        }
      }
      return n == 1 && o1 >= 0;
    };
    Index fo0 = 0;
    Index fo1 = 0;
    Index go0 = 0;
    Index go1 = 0;
    if (!opposite_edge(f, vi, fo0, fo1) || !opposite_edge(g, vi, go0, go1)) {
      return false;
    }
    if (fo0 == go0 || fo0 == go1 || fo1 == go0 || fo1 == go1) {
      return false;  // edge-adjacent (a shared second vertex); the finder skips these
    }
    return segment3_triangle3_intersect(pos(fo0), pos(fo1), pos(g(0)), pos(g(1)), pos(g(2))) ||
           segment3_triangle3_intersect(pos(go0), pos(go1), pos(f(0)), pos(f(1)), pos(f(2)));
  }

  // Whether bisecting edge e at midpoint m (new vertex nv; the split's two side faces fi0/fi1 with
  // apexes c/d) would create a self-intersection that MeshDefectsFinder counts. A midpoint split is
  // geometry-preserving, but it drops the shared-vertex count between overlapping faces, so an
  // overlap that was edge-adjacent (which the finder skips) can become the vertex-adjacent kind it
  // counts. This applies the finder's own criterion to the vertices whose star the split changes
  // (e.a, e.b, c, d, nv); every other star is unchanged and was already clean, so it matches the
  // finder's global verdict. Untransformed frame, matching the finder.
  static bool split_reveals_intersection(const AbstractMesh& am, const Points3& p, const Edge& e,
                                         Index c, Index d, Index nv, const Point3& m, Index fi0,
                                         Index fi1) {
    auto pos = [&](Index v) -> Point3 { return v == nv ? m : Point3(p.row(v)); };
    // the four sub-faces (with the new vertex nv), wound as insert_on_edge produces them
    std::array<Face, 4> subs{Face{e.a, nv, c}, Face{nv, e.b, c}, Face{e.b, nv, d},
                             Face{nv, e.a, d}};

    for (Index vi : {e.a, e.b, c, d, nv}) {
      // vi's post-split star: its committed faces minus the two split sides, plus the incident
      // sub-faces. Only pairs involving a sub-face are new; the rest were checked in an earlier
      // pass.
      std::vector<Face> existing;
      if (vi != nv) {
        for (auto fi : am.vertex_faces(vi)) {
          if (fi != fi0 && fi != fi1) {
            existing.push_back(am.face(fi));
          }
        }
      }
      std::vector<Face> incident_subs;
      for (const auto& s : subs) {
        if (s(0) == vi || s(1) == vi || s(2) == vi) {
          incident_subs.push_back(s);
        }
      }
      for (std::size_t i = 0; i < incident_subs.size(); i++) {
        for (const auto& g : existing) {
          if (finder_pair_hits(vi, incident_subs.at(i), g, pos)) {
            return true;
          }
        }
        for (std::size_t j = i + 1; j < incident_subs.size(); j++) {
          if (finder_pair_hits(vi, incident_subs.at(i), incident_subs.at(j), pos)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  // Whether collapsing a onto b (dropping a) would create a self-intersection MeshDefectsFinder
  // counts. Only the kept faces move (a retargeted to b); the finder's criterion is applied at each
  // vertex whose star they change -- b and every other kept-face vertex -- against that vertex's
  // unchanged faces and the other kept faces. Untransformed frame, matching the finder.
  bool collapse_reveals_intersection(const boost::unordered_flat_set<Index>& star,
                                     const std::vector<Face>& kept) const {
    auto pos = [&](Index v) -> Point3 { return p_.row(v); };
    boost::unordered_flat_set<Index> affected;
    for (const auto& nf : kept) {
      for (auto v : nf) {
        affected.insert(v);
      }
    }
    for (Index vi : affected) {
      std::vector<Face> existing;
      for (auto fi : mesh_.vertex_faces(vi)) {
        if (!star.contains(fi)) {
          existing.push_back(mesh_.face(fi));
        }
      }
      std::vector<Face> incident_kept;
      for (const auto& nf : kept) {
        if (nf(0) == vi || nf(1) == vi || nf(2) == vi) {
          incident_kept.push_back(nf);
        }
      }
      for (std::size_t i = 0; i < incident_kept.size(); i++) {
        for (const auto& g : existing) {
          if (finder_pair_hits(vi, incident_kept.at(i), g, pos)) {
            return true;
          }
        }
        for (std::size_t j = i + 1; j < incident_kept.size(); j++) {
          if (finder_pair_hits(vi, incident_kept.at(i), incident_kept.at(j), pos)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  // Whether collapsing v onto w is admissible; dev = the dropped point's distance to the new
  // surface.
  bool collapse_ok(Halfedge h, const std::vector<Halfedge>& hs, double& dev) {
    auto a = mesh_.from(h);  // the dropped vertex
    auto b = mesh_.to(h);    // the kept vertex
    auto c = mesh_.apex(h);
    auto d = mesh_.apex(mesh_.opposite(h));

    // The link condition: Lk(a) \cap Lk(b) =? Lk(a \cup b) = {c, d}.
    for (auto hh : hs) {
      auto v = mesh_.to(hh);
      if (v != b && v != c && v != d && mesh_.has_edge({v, b})) {
        // v \in Lk(a) \cap Lk(b).
        return false;
      }
    }

    // a's kept faces (those not on edge ab), with a retargeted to b.
    std::vector<Face> kept;
    boost::unordered_flat_set<Index> star;
    for (auto hh : hs) {
      auto fi = mesh_.face(hh);
      star.insert(fi);
      auto f = mesh_.face(fi);
      if (on_edge(f, a, b)) {
        continue;  // collapses to a degenerate sliver, dropped
      }
      Face nf = (f.array() == a).select(b, f);
      auto nn = normal(nf);
      if (!(nn.norm() > 0.0)) {
        return false;  // the face would become degenerate
      }
      if (nn.dot(normal(f)) <= 0.0) {
        return false;  // the face would flip
      }
      kept.push_back(nf);
    }
    if (kept.empty()) {
      return false;
    }

    // Cap edge length: refuse a collapse that stretches a neighbour edge past kMaxEdgeRatio * res.
    // Without it a collapse feeds a long edge to the split, whose apex-connecting bisection re-adds
    // a short edge, and the two churn without converging; the cap breaks that cycle. Relaxation
    // between calls un-wedges a sliver the cap has to defer.
    for (auto hh : hs) {
      auto v = mesh_.to(hh);
      if (v != b && v != c && v != d && (ap_.row(b) - ap_.row(v)).squaredNorm() > max_edge2_) {
        return false;
      }
    }

    // Distortion (for picking the least-distorting neighbour): the dropped vertex's distance to the
    // new surface.
    dev = std::numeric_limits<double>::infinity();
    for (const auto& nf : kept) {
      dev = std::min(dev, dist2(ap_.row(a), nf));
    }

    // Faces near the collapse: those incident to the kept faces' vertices but outside the star.
    boost::unordered_flat_set<Index> nearby;
    for (const auto& nf : kept) {
      for (auto v : nf) {
        for (auto fi : mesh_.vertex_faces(v)) {
          if (!star.contains(fi)) {
            nearby.insert(fi);
          }
        }
      }
    }

    if (!honors_ok(star, kept, nearby)) {
      return false;
    }

    // No new face may self-intersect another; a collapse only moves the kept faces, so a new
    // overlap involves one of them. Two guards with disjoint reach: against a topologically distant
    // sheet (no shared vertex) that the one-ring cannot see, a spatial broad-phase catches a kept
    // face pushed onto a spatially near one...
    for (const auto& nf : kept) {
      auto ps = p_(nf, kAll);
      Point3 lo = ps.colwise().minCoeff();
      Point3 hi = ps.colwise().maxCoeff();
      if (face_grid_.any_of(lo, hi, [&](Index fi) {
            auto g = mesh_.face(fi);
            return !star.contains(fi) && num_shared_vertices(nf, g) == 0 && intersects(nf, g);
          })) {
        return false;
      }
    }

    // ...and against a vertex-adjacent face, the finder's exact opposite-edge criterion catches a
    // fold (an edge-adjacent overlap turned vertex-adjacent) that the broad-phase's edge-skipping
    // test would miss. The broad-phase above therefore need only cover the no-shared-vertex case.
    if (collapse_reveals_intersection(star, kept)) {
      return false;
    }

    if (!single_fan_at(a, b, kept)) {
      return false;
    }

    return true;
  }

  // Whether b's incident faces after collapsing a onto it form a single fan -- i.e. b stays a
  // manifold vertex (matching MeshDefectsFinder: b's link graph is simple, connected, max-degree
  // 2). The link condition above admits one bad case it cannot see: if both a and b have a face on
  // the shared pair {c, d}, retargeting a's onto b doubles the triangle {b, c, d}, a fin whose
  // repeated link edge makes b singular. Near-touching sheets a fine lattice resolves are where
  // this arises.
  bool single_fan_at(Index a, Index b, const std::vector<Face>& kept) const {
    std::vector<std::array<Index, 2>> edges;  // each incident face's opposite edge (its two non-b
    auto add = [&](const Face& f) {           // vertices), the edges of b's link graph
      std::array<Index, 2> e{};
      auto j = 0;
      for (auto k = 0; k < 3; k++) {
        if (f(k) != b) {
          e.at(j++) = f(k);
        }
      }
      if (e.at(0) > e.at(1)) {
        std::swap(e.at(0), e.at(1));
      }
      edges.push_back(e);
    };
    for (auto fi : mesh_.vertex_faces(b)) {
      auto f = mesh_.face(fi);
      if ((f.array() == a).any()) {
        continue;  // a face on the collapsed edge ab, removed by the collapse
      }
      add(f);
    }
    for (const auto& nf : kept) {
      add(nf);  // a's faces, retargeted onto b
    }
    if (edges.empty()) {
      return true;
    }

    std::vector<Index> verts;
    for (const auto& e : edges) {
      for (auto v : e) {
        if (std::ranges::find(verts, v) == verts.end()) {
          verts.push_back(v);
        }
      }
    }
    auto idx = [&](Index v) {
      return static_cast<Index>(std::ranges::find(verts, v) - verts.begin());
    };

    // Simple (no doubled link edge -> no fin) and max-degree 2 (no non-manifold edge at b).
    std::vector<Index> degree(verts.size(), 0);
    for (std::size_t i = 0; i < edges.size(); i++) {
      if (edges.at(i).at(0) == edges.at(i).at(1)) {
        return false;  // degenerate face
      }
      for (auto j = i + 1; j < edges.size(); j++) {
        if (edges.at(i) == edges.at(j)) {
          return false;  // doubled link edge -> b is singular
        }
      }
      degree.at(idx(edges.at(i).at(0)))++;
      degree.at(idx(edges.at(i).at(1)))++;
    }
    if (std::ranges::any_of(degree, [](Index dgr) { return dgr > 2; })) {
      return false;
    }

    // Connected (a single fan, not two meeting only at b).
    std::vector<Index> parent(verts.size());
    std::iota(parent.begin(), parent.end(), Index{0});
    auto root = [&](Index x) {
      while (parent.at(x) != x) {
        parent.at(x) = parent.at(parent.at(x));
        x = parent.at(x);
      }
      return x;
    };
    for (const auto& e : edges) {
      auto r0 = root(idx(e.at(0)));
      auto r1 = root(idx(e.at(1)));
      if (r0 != r1) {
        parent.at(r0) = r1;
      }
    }
    Index components = 0;
    for (Index i = 0; i < static_cast<Index>(verts.size()); i++) {
      if (root(i) == i) {
        components++;
      }
    }
    return components <= 1;
  }

  double dist2(const Point3& p, const Face& f) const {
    return point_triangle_dist2(p, ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  Mesh emit() {
    Points3 vertices(p_.rows(), 3);
    auto faces = std::move(mesh_).take_faces();
    Index nv = 0;
    std::vector<Index> vv(p_.rows(), -1);
    for (auto f : faces.rowwise()) {
      for (auto k = 0; k < 3; k++) {
        auto v = f(k);
        if (vv.at(v) < 0) {
          vv.at(v) = nv;
          vertices.row(nv) = p_.row(v);
          nv++;
        }
        f(k) = vv.at(v);
      }
    }
    vertices.conservativeResize(nv, Eigen::NoChange);
    return {std::move(vertices), std::move(faces)};
  }

  // Whether snap point i lies within its tolerance of face f.
  bool honored_by(Index i, const Face& f) const {
    return dist2(a_points_.row(i), f) <= snap_tols2_(i);
  }

  // Every nearby snap point held by a removed face -- not just the dropped vertex -- must stay
  // within tolerance, else a greedy chain drifts already-dropped points off the surface.
  bool honors_ok(const boost::unordered_flat_set<Index>& star, const std::vector<Face>& kept,
                 const boost::unordered_flat_set<Index>& nearby) const {
    if (snap_grid_.empty()) {
      return true;
    }
    auto face_of = [&](Index fi) -> Face { return mesh_.face(fi); };
    Point3 lo = ap_.row(face_of(*star.begin())(0));
    Point3 hi = lo;
    for (auto fi : star) {
      for (auto x : face_of(fi)) {
        lo = lo.cwiseMin(ap_.row(x));
        hi = hi.cwiseMax(ap_.row(x));
      }
    }
    bool ok = true;
    snap_grid_.for_each(lo, hi, [&](Index i) {
      auto honored = [&](const auto& f) { return honored_by(i, f); };
      if (std::ranges::none_of(star, honored, face_of)) {
        return true;  // not held by a removed face; the collapse cannot dishonor it
      }
      if (std::ranges::none_of(kept, honored) && std::ranges::none_of(nearby, honored, face_of)) {
        ok = false;
        return false;  // dishonored; stop the walk
      }
      return true;
    });
    return ok;
  }

  void index_face(Index fi) { face_grid_.insert(fi, p_(mesh_.face(fi), kAll)); }

  // The self-intersection guard runs in the untransformed frame (p_), where defects are judged,
  // matching the defect finder.
  bool intersects(const Face& a, const Face& b) const {
    return triangles_intersect(p_.row(a(0)), p_.row(a(1)), p_.row(a(2)), p_.row(b(0)), p_.row(b(1)),
                               p_.row(b(2)), num_shared_vertices(a, b));
  }

  Vector3 normal(const Face& f) const {
    return triangle_normal(ap_.row(f(0)), ap_.row(f(1)), ap_.row(f(2)));
  }

  static bool on_edge(const Face& f, Index a, Index b) {
    return (f.array() == a).any() && (f.array() == b).any();
  }

  // Collapse v onto its least-distorting admissible neighbour, if any; returns the kept vertex, or
  // -1 if none was admissible. Distortion (the dropped vertex's distance to the new surface)
  // already keeps feature vertices -- dropping one distorts the surface most, so it loses to a
  // smoother neighbour.
  Index try_collapse(Index v) {
    auto out = mesh_.vertex_outgoing_halfedges(v);
    std::vector<Halfedge> hs(out.begin(), out.end());  // copy: collapse rewrites the adjacency
    if (hs.size() < 3) {
      return -1;
    }
    // hs holds v's outgoing halfedges v -> w. v must be an interior manifold vertex: each one's
    // opposite (w -> v) must also have a face.
    if (std::ranges::any_of(hs, [&](Halfedge h) { return !mesh_.opposite(h).is_valid(); })) {
      return -1;
    }

    Halfedge best{};  // the chosen collapse v -> w
    double best_dev = std::numeric_limits<double>::infinity();
    for (auto h : hs) {
      double dev = 0.0;
      if (collapse_ok(h, hs, dev) && dev < best_dev) {
        best = h;
        best_dev = dev;
      }
    }
    if (!best.is_valid()) {
      return -1;
    }
    Index kept = mesh_.to(best);  // capture before collapse invalidates the halfedge
    // Drop the star from the grid before the collapse rewrites its faces, then re-add the
    // retargeted faces.
    for (auto h : hs) {
      unindex_face(mesh_.face(h));
    }
    for (auto fi : mesh_.collapse(best)) {  // drop best.from onto best.to
      index_face(fi);
    }
    return kept;
  }

  void unindex_face(Index fi) { face_grid_.remove(fi); }

  Points3 p_;
  Points3 ap_;
  AbstractMesh mesh_;  // working connectivity, edited in place by collapses
  Points3 a_points_;   // the snap targets
  VecX snap_tols2_;    // squared snapping tolerance per snap point
  SpatialGrid snap_grid_;
  FaceGrid face_grid_;
  double max_edge2_{};  // above this squared length an edge is split
  double min_edge2_{};  // below this squared length an edge is a sliver to collapse
  Mesh result_;
};

}  // namespace polatory::isosurface::snapper
