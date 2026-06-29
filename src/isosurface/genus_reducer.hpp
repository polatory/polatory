#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <optional>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/neighbor.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dense_undirected_graph.hpp"
#include "disjoint_sets.hpp"
#include "indexer.hpp"
#include "snapper/utility.hpp"

namespace polatory::isosurface {

// Removes sub-resolution topological artifacts from the RMT surface. A vertex set whose link -- the
// star edges opposite it -- is two or more disjoint cycles forms an annulus; capping each cycle
// severs it. Some artifacts make a single node's link an annulus; others appear only on the union
// of two adjacent nodes. Cuts share no face, so they are independent and need no rollback.
class GenusReducer {
  using LatticeCoordinates = rmt::LatticeCoordinates;
  using LatticeCoordinatesHash = rmt::LatticeCoordinatesHash;
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using PrimitiveLattice = rmt::PrimitiveLattice;

  // An annulus region, carrying everything commit needs to cut it.
  struct Candidate {
    std::unordered_set<Index> vs;           // the region's vertices
    std::unordered_set<Index> star;         // the star face indices
    std::vector<std::vector<Index>> loops;  // per link cycle, its link vertices
  };

 public:
  GenusReducer(const Mesh& mesh, const PrimitiveLattice& lattice)
      : v_(mesh.vertices()),
        f_(mesh.faces()),
        nv_(v_.rows()),
        nf_(f_.rows()),
        vf_(nv_),
        deleted_(nf_, false) {
    for (Index fi = 0; fi < nf_; fi++) {
      for (auto k = 0; k < 3; k++) {
        vf_.at(f_(fi, k)).push_back(fi);
      }
    }

    std::unordered_map<LatticeCoordinates, std::unordered_set<Index>, LatticeCoordinatesHash>
        node_vs;
    for (Index v = 0; v < nv_; v++) {
      node_vs[lattice.lattice_coordinates_rounded(v_.row(v))].insert(v);
    }

    // Vertices of every node whose own link is an annulus.
    std::unordered_set<Index> lone_vs;
    for (const auto& [node, vertices] : node_vs) {
      if (analyze(vertices)) {
        lone_vs.insert(vertices.begin(), vertices.end());
      }
    }

    // Vertices of every adjacent node pair whose combined link is an annulus though neither node's
    // is. Kept apart from lone_vs so they cannot merge a lone-node annulus into a non-annulus blob.
    std::unordered_set<Index> pair_vs;
    for (const auto& [lc, verts] : node_vs) {
      for (rmt::EdgeIndex ei = 0; ei < 14; ei++) {
        auto nlc = rmt::neighbor(lc, ei);
        auto it = node_vs.find(nlc);
        if (it == node_vs.end() || rmt::LatticeCoordinatesLess()(nlc, lc)) {
          continue;
        }
        std::unordered_set<Index> u = verts;
        u.insert(it->second.begin(), it->second.end());
        if (analyze(u)) {
          pair_vs.insert(u.begin(), u.end());
        }
      }
    }

    // Cut each set's components, lone-node first; a component is one artifact, spanning any number
    // of nodes. A pair component whose star an earlier cut already touched is skipped, keeping cuts
    // independent.
    for (const auto& component : connected_components(lone_vs)) {
      if (auto candidate = analyze(component)) {
        commit(*candidate);
      }
    }
    for (const auto& component : connected_components(pair_vs)) {
      if (auto candidate = analyze(component)) {
        if (std::ranges::none_of(candidate->star, [this](Index fi) { return deleted_.at(fi); })) {
          commit(*candidate);
        }
      }
    }

    result_ = emit();
  }

  Mesh result() && { return std::move(result_); }

 private:
  // Tests whether the vertex set vs is an annulus -- its link, the star edges opposite vs, is two
  // or more disjoint simple cycles rather than the single cycle bounding an ordinary disk --
  // reading the star off the original mesh so the verdict is independent of any cut. Returns the
  // data to cut it, or nullopt.
  std::optional<Candidate> analyze(const std::unordered_set<Index>& vs) const {
    if (vs.size() < 2) {
      return std::nullopt;
    }

    std::unordered_set<Index> star;
    for (auto v : vs) {
      for (auto fi : vf_.at(v)) {
        star.insert(fi);
      }
    }

    // The link -- the star edges opposite vs -- as a graph over the link vertices.
    ValueIndexer<Index> link_vertices;
    std::vector<Edge> link_edges;
    for (auto fi : star) {
      if (auto e = link_edge(fi, vs)) {
        link_vertices.insert(e->a);
        link_vertices.insert(e->b);
        link_edges.push_back(*e);
      }
    }
    if (link_edges.empty()) {
      return std::nullopt;
    }
    DenseUndirectedGraph link(std::move(link_vertices));
    for (const auto& e : link_edges) {
      link.add_edge(e.a, e.b);
    }

    // is_simple rejects a doubled edge (two star faces sharing one opposite edge, which degree
    // alone reads as a 2-cycle); the degree bounds reject boundary nodes (a degree-1 open path).
    if (!(link.is_simple() && link.min_degree() == 2 && link.max_degree() == 2)) {
      return std::nullopt;
    }
    auto loops = link.connected_components();  // each cycle's link vertices
    if (loops.size() < 2) {
      return std::nullopt;  // a single disk boundary -- not an annulus
    }

    Candidate candidate;
    candidate.vs = vs;
    candidate.star = std::move(star);
    candidate.loops = std::move(loops);
    return candidate;
  }

  // Whether the cut's caps can be emitted without a defect: each cap is non-degenerate, no two
  // cycles' apexes coincide (which would pinch the surface to a point), and no cap crosses a cap of
  // another cycle or a retained face it borders. A cycle's own caps form a fan sharing its apex;
  // that fan is left to the on-surface apex placement and not re-checked here.
  bool caps_ok(const std::vector<Point3>& apex_pos,
               const std::vector<std::pair<Index, Index>>& caps,
               const std::unordered_set<Index>& retained,
               const std::unordered_set<Index>& vs) const {
    struct Cap {
      Point3 p;  // the apex
      Point3 q;  // the link vertex j
      Point3 r;  // the link vertex k
      Index j;
      Index k;
      Index cycle;
    };
    std::vector<Cap> tris;
    tris.reserve(caps.size());
    double scale = 0.0;  // local feature size, for the degeneracy/coincidence tolerances
    for (const auto& [c, fi] : caps) {
      auto e = link_edge(fi, vs).value();
      Cap cap{
          .p = apex_pos.at(c), .q = v_.row(e.a), .r = v_.row(e.b), .j = e.a, .k = e.b, .cycle = c};
      scale =
          std::max({scale, (cap.q - cap.p).norm(), (cap.r - cap.p).norm(), (cap.r - cap.q).norm()});
      tris.push_back(cap);
    }
    if (scale == 0.0) {
      return false;
    }
    for (const auto& t : tris) {
      if ((t.q - t.p).cross(t.r - t.p).norm() <= 1e-9 * scale * scale) {
        return false;  // a degenerate (near-zero-area) cap
      }
    }
    for (Index a = 0; a < static_cast<Index>(apex_pos.size()); a++) {
      for (Index b = a + 1; b < static_cast<Index>(apex_pos.size()); b++) {
        if ((apex_pos.at(a) - apex_pos.at(b)).norm() <= 1e-6 * scale) {
          return false;  // coincident apexes -> pinch
        }
      }
    }
    for (std::size_t i = 0; i < tris.size(); i++) {
      for (std::size_t j = i + 1; j < tris.size(); j++) {
        const auto& a = tris.at(i);
        const auto& b = tris.at(j);
        if (a.cycle != b.cycle &&  // cross-cycle caps share no vertex
            snapper::triangles_intersect(a.p, a.q, a.r, b.p, b.q, b.r, 0)) {
          return false;
        }
      }
    }
    for (const auto& cap : tris) {
      for (auto fi : retained) {
        auto shared = 0;
        for (auto m = 0; m < 3; m++) {
          if (f_(fi, m) == cap.j || f_(fi, m) == cap.k) {
            shared++;
          }
        }
        if (snapper::triangles_intersect(cap.p, cap.q, cap.r, v_.row(f_(fi, 0)), v_.row(f_(fi, 1)),
                                         v_.row(f_(fi, 2)), shared)) {
          return false;
        }
      }
    }
    return true;
  }

  // Cuts the region: one cap per link cycle, its apex placed two thirds of the way from the opening
  // toward the region centroid, so the caps seal the openings from inside. A cut whose caps would
  // self-intersect is dropped -- no rollback follows -- leaving the region uncut.
  void commit(const Candidate& candidate) {
    const auto& vs = candidate.vs;
    const auto& loops = candidate.loops;
    auto n_cycles = static_cast<Index>(loops.size());

    // Each link vertex -> its cycle, for routing each cap to its cycle's apex.
    std::unordered_map<Index, Index> cycle_of;
    for (Index c = 0; c < n_cycles; c++) {
      for (auto v : loops.at(c)) {
        cycle_of.emplace(v, c);
      }
    }

    // Each link-edge star face becomes a cap (its node vertex -> the apex of that edge's cycle).
    std::vector<std::pair<Index, Index>> caps;  // (cycle, star face)
    for (auto fi : candidate.star) {
      if (auto e = link_edge(fi, vs)) {
        caps.emplace_back(cycle_of.at(e->a), fi);
      }
    }

    // The region centroid, on the inner side of every opening.
    Point3 region_centroid = Point3::Zero();
    for (auto v : vs) {
      region_centroid += v_.row(v);
    }
    region_centroid /= static_cast<double>(vs.size());

    std::vector<Point3> apex_pos(n_cycles);
    std::unordered_set<Index> retained;  // faces around the loops, kept (the caps continue them)
    for (Index c = 0; c < n_cycles; c++) {
      const auto& loop = loops.at(c);
      for (auto w : loop) {
        for (auto fi : vf_.at(w)) {
          if (!candidate.star.contains(fi)) {
            retained.insert(fi);
          }
        }
      }
      // Two thirds of the way from the opening toward the region centroid.
      Point3 loop_centroid = Point3::Zero();
      for (auto w : loop) {
        loop_centroid += v_.row(w);
      }
      loop_centroid /= static_cast<double>(loop.size());
      apex_pos.at(c) = (1.0 / 3.0) * loop_centroid + (2.0 / 3.0) * region_centroid;
    }

    if (!caps_ok(apex_pos, caps, retained, vs)) {
      return;
    }

    // Add the apex vertices, drop the star, and emit the caps (winding preserved by the
    // node-vertex -> apex substitution).
    std::vector<Index> apex(n_cycles);
    for (Index c = 0; c < n_cycles; c++) {
      apex.at(c) = nv_ + static_cast<Index>(new_positions_.size());
      new_positions_.push_back(apex_pos.at(c));
    }
    for (auto fi : candidate.star) {
      deleted_.at(fi) = true;
    }
    for (const auto& [c, fi] : caps) {
      Face cap;
      for (auto k = 0; k < 3; k++) {
        cap(k) = vs.contains(f_(fi, k)) ? apex.at(c) : f_(fi, k);
      }
      new_faces_.push_back(cap);
    }
  }

  // Splits a vertex set into its connected components over mesh edges joining two of its vertices.
  std::vector<std::unordered_set<Index>> connected_components(
      const std::unordered_set<Index>& vs) const {
    DisjointSets sets(ValueIndexer<Index>{vs});
    for (auto v : vs) {
      for (auto fi : vf_.at(v)) {
        for (auto k = 0; k < 3; k++) {
          Index w = f_(fi, k);
          if (w != v && vs.contains(w)) {
            sets.unite(v, w);
          }
        }
      }
    }
    std::vector<std::unordered_set<Index>> components;
    for (auto& members : sets.groups()) {
      components.emplace_back(members.begin(), members.end());
    }
    return components;
  }

  Mesh emit() const {
    auto total = nv_ + static_cast<Index>(new_positions_.size());
    std::vector<Face> faces;
    for (Index fi = 0; fi < nf_; fi++) {
      if (!deleted_.at(fi)) {
        faces.push_back(f_.row(fi));
      }
    }
    faces.insert(faces.end(), new_faces_.begin(), new_faces_.end());

    std::vector<bool> used(total, false);
    for (const auto& f : faces) {
      for (auto v : f) {
        used.at(v) = true;
      }
    }
    std::vector<Index> vv(total, -1);
    Index n = 0;
    for (Index v = 0; v < total; v++) {
      if (used.at(v)) {
        vv.at(v) = n++;
      }
    }
    Points3 vertices(n, 3);
    for (Index v = 0; v < total; v++) {
      if (used.at(v)) {
        vertices.row(vv.at(v)) = v < nv_ ? v_.row(v) : new_positions_.at(v - nv_);
      }
    }
    Faces out(static_cast<Index>(faces.size()), 3);
    for (Index i = 0; i < out.rows(); i++) {
      out.row(i) = Face{vv.at(faces.at(i)(0)), vv.at(faces.at(i)(1)), vv.at(faces.at(i)(2))};
    }
    return {std::move(vertices), std::move(out)};
  }

  // The link edge of star face fi -- its two corners that are not the node's -- or nullopt if the
  // face shares more than one vertex with the node, leaving fewer than two such corners. The edge
  // is unordered: its endpoints carry no winding.
  std::optional<Edge> link_edge(Index fi, const std::unordered_set<Index>& vs) const {
    std::array<Index, 2> e{};
    auto n = 0;
    for (auto k = 0; k < 3; k++) {
      if (!vs.contains(f_(fi, k))) {
        if (n == 2) {
          return std::nullopt;
        }
        e.at(n++) = f_(fi, k);
      }
    }
    if (n != 2) {
      return std::nullopt;
    }
    return Edge{e.at(0), e.at(1)};
  }

  Points3 v_;
  Faces f_;
  Index nv_;
  Index nf_;
  std::vector<std::vector<Index>> vf_;
  std::vector<bool> deleted_;          // original faces removed by a cut
  std::vector<Point3> new_positions_;  // cap apexes, indexed from nv_
  std::vector<Face> new_faces_;        // cap triangles
  Mesh result_;
};

}  // namespace polatory::isosurface
