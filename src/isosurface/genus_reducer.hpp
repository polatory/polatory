#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <optional>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/dense_undirected_graph.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "quadric_position.hpp"
#include "snapper/utility.hpp"

namespace polatory::isosurface {

// Cuts sub-resolution tunnels from the RMT surface. A lattice node whose surface star is an annulus
// -- its link (the edges opposite the node) forms >= 2 disjoint cycles instead of one disk boundary
// -- straddles a tunnel pinched below the resolution. Replacing the star with one cap per link
// cycle severs the tunnel and seals each opening, dropping the genus while changing only that
// node's incident faces; a clean single-disk node is left untouched.
class GenusReducer {
  using LatticeCoordinates = rmt::LatticeCoordinates;
  using LatticeCoordinatesHash = rmt::LatticeCoordinatesHash;
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using PrimitiveLattice = rmt::PrimitiveLattice;

  // An annulus node found in pass 1, carrying everything pass 2 needs to cut it. star_vs (the
  // node's vertices plus the link vertices) are what neighboring cuts are tested against.
  struct Candidate {
    LatticeCoordinates node{};
    std::unordered_set<Index> vs;               // the node's vertices
    std::vector<Index> star;                    // the star face indices
    std::unordered_map<Index, Index> cycle_of;  // each link vertex -> its link cycle
    Index n_cycles = 0;
    std::vector<Index> star_vs;  // the node's vertices + link vertices, for disjointness
  };

 public:
  GenusReducer(const Mesh& mesh, const PrimitiveLattice& lattice, const Mat3& aniso)
      : v_(mesh.vertices()),
        f_(mesh.faces()),
        nv_(v_.rows()),
        nf_(f_.rows()),
        vf_(nv_),
        lattice_(lattice),
        aniso_(aniso),
        aniso_inv_(aniso.inverse()),
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

    // Pass 1: find every annulus node. Each node's star is read off the original mesh
    // independently, so the set of cuts does not depend on the order nodes are visited. Each cut
    // claims its star's vertices.
    std::vector<Candidate> candidates;
    std::unordered_map<Index, int> claims;
    for (auto& [node, vs] : node_vs) {
      auto candidate = analyze(node, std::move(vs));
      if (candidate) {
        for (auto v : candidate->star_vs) {
          claims[v]++;
        }
        candidates.push_back(std::move(*candidate));
      }
    }

    // Pass 2: cut only annulus nodes whose star is vertex-disjoint from every other annulus node.
    // Such cuts touch disjoint faces, so they are mutually independent and their combined result is
    // the same in any order. Two adjacent annuli -- which could leave a shared vertex non-manifold
    // if both were cut, with no rollback in this pass to catch it -- are both left uncut.
    for (const auto& candidate : candidates) {
      if (std::ranges::all_of(candidate.star_vs, [&](Index v) { return claims.at(v) == 1; })) {
        commit(candidate);
      }
    }

    result_ = emit();
  }

  Mesh result() && { return std::move(result_); }

 private:
  // Tests whether node is an annulus -- its link is two or more disjoint simple cycles rather than
  // the single cycle bounding an ordinary disk -- reading the star off the original mesh so the
  // verdict is independent of any other cut. Returns the data to cut it, or nullopt.
  std::optional<Candidate> analyze(const LatticeCoordinates& node,
                                   std::unordered_set<Index> vs) const {
    if (vs.size() < 2) {
      return std::nullopt;
    }

    std::vector<Index> star;
    std::unordered_set<Index> star_set;
    for (auto v : vs) {
      for (auto fi : vf_.at(v)) {
        if (star_set.insert(fi).second) {
          star.push_back(fi);
        }
      }
    }

    // The link -- the star edges opposite the node -- as a graph over a local vertex numbering.
    std::unordered_map<Index, Index> to_local;
    std::vector<Index> from_local;
    auto local = [&](Index v) {
      auto [it, inserted] = to_local.try_emplace(v, static_cast<Index>(from_local.size()));
      if (inserted) {
        from_local.push_back(v);
      }
      return it->second;
    };
    std::vector<std::array<Index, 2>> link_edges;
    for (auto fi : star) {
      if (auto e = link_edge(fi, vs)) {
        link_edges.push_back({local(e->a), local(e->b)});
      }
    }
    if (link_edges.empty()) {
      return std::nullopt;
    }
    DenseUndirectedGraph link(static_cast<Index>(from_local.size()));
    for (const auto& e : link_edges) {
      link.add_edge(e.at(0), e.at(1));
    }

    // is_simple rejects a doubled edge (two star faces sharing one opposite edge, which degree
    // alone reads as a 2-cycle); the degree bounds reject boundary nodes (a degree-1 open path).
    if (!(link.is_simple() && link.min_degree() == 2 && link.max_degree() == 2)) {
      return std::nullopt;
    }
    auto cycle = link.connected_components();
    auto n_cycles = *std::ranges::max_element(cycle) + 1;
    if (n_cycles < 2) {
      return std::nullopt;  // a single disk boundary -- an ordinary node, not a tunnel
    }

    Candidate candidate;
    candidate.node = node;
    candidate.star = star;
    candidate.n_cycles = n_cycles;
    for (Index i = 0; i < static_cast<Index>(from_local.size()); i++) {
      candidate.cycle_of.emplace(from_local.at(i), cycle.at(i));
    }
    candidate.star_vs.assign(vs.begin(), vs.end());
    candidate.star_vs.insert(candidate.star_vs.end(), from_local.begin(), from_local.end());
    candidate.vs = std::move(vs);
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

  // Cuts an isolated annulus node: one apex per link cycle, placed by the area-weighted quadric fit
  // of the retained faces around the loop (the same placement vertex clustering uses), so each cap
  // lands on the surrounding surface and the caps separate to opposite tunnel sides. A cut whose
  // caps are not emittable is dropped (no rollback follows this pass), leaving the tunnel.
  void commit(const Candidate& candidate) {
    const auto& vs = candidate.vs;
    auto n_cycles = candidate.n_cycles;
    std::unordered_set<Index> star_set(candidate.star.begin(), candidate.star.end());

    std::vector<std::vector<Index>> loops(n_cycles);
    for (const auto& [v, c] : candidate.cycle_of) {
      loops.at(c).push_back(v);
    }

    std::vector<Point3> apex_pos(n_cycles);
    std::unordered_set<Index> retained;  // faces around the loops, kept (the caps continue them)
    for (Index c = 0; c < n_cycles; c++) {
      const auto& loop = loops.at(c);
      std::unordered_set<Index> cap_faces;
      for (auto w : loop) {
        for (auto fi : vf_.at(w)) {
          if (!star_set.contains(fi)) {
            cap_faces.insert(fi);
          }
        }
      }
      retained.insert(cap_faces.begin(), cap_faces.end());
      Points3 anchor(static_cast<Index>(loop.size()), 3);
      for (Index j = 0; j < static_cast<Index>(loop.size()); j++) {
        anchor.row(j) = v_.row(loop.at(j));
      }
      std::vector<std::array<Point3, 3>> triangles;
      triangles.reserve(cap_faces.size());
      for (auto fi : cap_faces) {
        triangles.push_back({v_.row(f_(fi, 0)), v_.row(f_(fi, 1)), v_.row(f_(fi, 2))});
      }
      apex_pos.at(c) =
          quadric_position(anchor, triangles, aniso_, aniso_inv_, lattice_, candidate.node);
    }

    // Each link-edge star face becomes a cap (its node vertex -> the apex of that edge's cycle).
    std::vector<std::pair<Index, Index>> caps;  // (cycle, star face)
    for (auto fi : candidate.star) {
      if (auto e = link_edge(fi, vs)) {
        caps.emplace_back(candidate.cycle_of.at(e->a), fi);
      }
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
  const PrimitiveLattice& lattice_;
  Mat3 aniso_;
  Mat3 aniso_inv_;
  std::vector<bool> deleted_;          // original faces removed by a cut
  std::vector<Point3> new_positions_;  // cap apexes, indexed from nv_
  std::vector<Face> new_faces_;        // cap triangles
  Mesh result_;
};

}  // namespace polatory::isosurface
