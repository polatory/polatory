#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dense_undirected_graph.hpp"
#include "disjoint_sets.hpp"
#include "indexer.hpp"
#include "quadric_position.hpp"

namespace polatory::isosurface {

// Clusters the RMT surface as a standalone mesh step, so it can be re-run between smoothing passes.
// Each lattice node's vertices are split into connected components, and a component is merged into
// one vertex (a quotient) only where merging stays manifold on the CURRENT mesh, so smoothing-
// induced connectivity changes make more nodes mergeable. That per-component test is necessary but
// not sufficient globally, so a detect-and-uncluster loop forbids any cluster whose merged vertex a
// defect finder flags, until the result is manifold.
class VertexClusterer {
  using LatticeCoordinates = rmt::LatticeCoordinates;
  using Point3 = geometry::Point3;
  using Points3 = geometry::Points3;
  using PrimitiveLattice = rmt::PrimitiveLattice;

  struct Cluster {
    std::vector<Index> vertices;
    Index rep;             // the min-index member, kept as the single merged vertex
    Point3 position;       // where the merged vertex is placed
    bool deleted = false;  // dropped after its merged vertex caused a defect
  };

 public:
  VertexClusterer(const Mesh& mesh, const PrimitiveLattice& lattice, const Mat3& aniso)
      : v_(mesh.vertices()),
        f_(mesh.faces()),
        aniso_(aniso),
        aniso_inv_(aniso.inverse()),
        nv_(v_.rows()),
        nf_(f_.rows()),
        vf_(nv_) {
    for (Index fi = 0; fi < nf_; fi++) {
      for (auto k = 0; k < 3; k++) {
        vf_.at(f_(fi, k)).push_back(fi);
      }
    }

    std::vector<LatticeCoordinates> vertex_node(nv_);
    for (Index v = 0; v < nv_; v++) {
      vertex_node.at(v) = lattice.lattice_coordinates_rounded(v_.row(v));
    }

    // Split each node's vertices into pieces connected by mesh edges within the node.
    DisjointSets sets(IdentityIndexer{nv_});
    for (auto f : f_.rowwise()) {
      for (auto k = 0; k < 3; k++) {
        Index v0 = f(k);
        Index v1 = f((k + 1) % 3);
        if (vertex_node.at(v0) == vertex_node.at(v1)) {
          sets.unite(v0, v1);
        }
      }
    }
    auto components = sets.groups();

    rep_.resize(nv_);
    std::iota(rep_.begin(), rep_.end(), Index{0});
    pos_ = v_;

    const auto& lo = lattice.first_extended_bbox().min();
    const auto& hi = lattice.first_extended_bbox().max();
    for (auto& component : components) {
      if (component.size() < 2) {
        continue;
      }

      auto v = component.front();
      auto p = lattice.position(vertex_node.at(v));
      if ((p.array() <= lo.array() || p.array() >= hi.array()).any()) {
        // Do not cluster vertices of a node on/outside the bbox, as this can produce a boundary
        // vertex inside the bbox.
        continue;
      }

      if (can_cluster(component)) {
        Index rep = *std::min_element(component.begin(), component.end());
        Point3 position = clustered_position(component, lattice, vertex_node.at(rep));
        clusters_.emplace_back(std::move(component), rep, position);
        register_cluster(clusters_.back());
      }
    }

    result_ = cluster();
  }

  Mesh result() && { return std::move(result_); }

 private:
  // Merging the component keeps the mesh manifold iff its link -- the outer edges of its incident
  // faces -- forms a single cycle (the component is a disk).
  bool can_cluster(const std::vector<Index>& component) const {
    std::unordered_set<Index> in_component(component.begin(), component.end());
    std::unordered_set<Index> faces;
    for (auto v : component) {
      for (auto fi : vf_.at(v)) {
        faces.insert(fi);
      }
    }
    // The link -- the star edges opposite the component -- as a graph over the link vertices.
    ValueIndexer<Index> link_vertices;
    std::vector<Edge> link_edges;
    for (auto fi : faces) {
      std::array<Index, 2> out{};
      auto n_out = 0;
      for (auto k = 0; k < 3; k++) {
        if (!in_component.contains(f_(fi, k))) {
          if (n_out < 2) {
            out.at(n_out) = f_(fi, k);
          }
          n_out++;
        }
      }
      if (n_out == 2) {
        link_vertices.insert(out.at(0));
        link_vertices.insert(out.at(1));
        link_edges.emplace_back(out.at(0), out.at(1));
      }
    }
    if (link_edges.empty()) {
      return false;
    }

    DenseUndirectedGraph link(std::move(link_vertices));
    for (const auto& e : link_edges) {
      link.add_edge(e.a, e.b);
    }

    // A disk: the link is a single cycle -- connected and 2-regular, with at least three vertices.
    return link.order() >= 3 && link.is_connected() && link.min_degree() == 2 &&
           link.max_degree() == 2;
  }

  Mesh cluster() {
    while (true) {
      auto [mesh, vertex_ci] = clustered_mesh();
      MeshDefectsFinder defects(mesh);
      auto vis = defects.singular_vertices();
      auto fis = defects.intersecting_faces();
      std::unordered_set<Index> flagged(vis.begin(), vis.end());
      for (auto fi : fis) {
        for (auto k = 0; k < 3; k++) {
          flagged.insert(mesh.faces()(fi, k));
        }
      }
      bool changed = false;
      for (auto v : flagged) {
        auto it = vertex_ci.find(v);
        if (it == vertex_ci.end()) {
          continue;
        }
        auto& cluster = clusters_.at(it->second);
        if (cluster.deleted) {
          continue;
        }
        unregister_cluster(cluster);
        cluster.deleted = true;
        changed = true;
      }
      if (!changed) {
        return mesh;
      }
    }
  }

  // Second return maps each merged output vertex to the cluster it came from.
  std::pair<Mesh, std::unordered_map<Index, std::size_t>> clustered_mesh() const {
    Index out_nv = 0;
    Index out_nf = 0;

    // Each vertex maps to the output index of its representative (reps numbered in index order).
    std::vector<Index> vv(nv_, -1);
    for (Index v = 0; v < nv_; v++) {
      Index r = rep_.at(v);
      if (vv.at(r) < 0) {
        vv.at(r) = out_nv++;
      }
      vv.at(v) = vv.at(r);
    }

    Points3 out_vertices(out_nv, 3);
    for (Index v = 0; v < nv_; v++) {
      if (rep_.at(v) == v) {
        out_vertices.row(vv.at(v)) = pos_.row(v);
      }
    }

    Faces out_faces(nf_, 3);
    for (auto f : f_.rowwise()) {
      Index v0 = vv.at(f(0));
      Index v1 = vv.at(f(1));
      Index v2 = vv.at(f(2));
      if (v0 == v1 || v1 == v2 || v2 == v0) {
        continue;
      }
      out_faces.row(out_nf++) = Face{v0, v1, v2};
    }
    out_faces.conservativeResize(out_nf, 3);

    std::unordered_map<Index, std::size_t> vertex_ci;
    for (std::size_t ci = 0; ci < clusters_.size(); ci++) {
      const auto& cluster = clusters_.at(ci);
      if (cluster.deleted) {
        continue;
      }
      vertex_ci[vv.at(cluster.rep)] = ci;
    }

    return {Mesh{std::move(out_vertices), std::move(out_faces)}, vertex_ci};
  }

  // Merges the cluster's vertices to the quadric minimizer over its incident face planes (see
  // quadric_position), keeping a crease/corner instead of averaging it away.
  Point3 clustered_position(const std::vector<Index>& cluster, const PrimitiveLattice& lattice,
                            const LatticeCoordinates& node) const {
    std::unordered_set<Index> fis;
    for (auto v : cluster) {
      for (auto fi : vf_.at(v)) {
        fis.insert(fi);
      }
    }
    std::vector<std::array<Point3, 3>> triangles;
    triangles.reserve(fis.size());
    for (auto fi : fis) {
      triangles.push_back({v_.row(f_(fi, 0)), v_.row(f_(fi, 1)), v_.row(f_(fi, 2))});
    }
    return quadric_position(v_(cluster, kAll), triangles, aniso_, aniso_inv_, lattice, node);
  }

  // Apply a cluster's merge to rep_/pos_, or undo it.
  void register_cluster(const Cluster& cluster) {
    pos_.row(cluster.rep) = cluster.position;
    for (auto v : cluster.vertices) {
      rep_.at(v) = cluster.rep;
    }
  }

  void unregister_cluster(const Cluster& cluster) {
    pos_.row(cluster.rep) = v_.row(cluster.rep);
    for (auto v : cluster.vertices) {
      rep_.at(v) = v;
    }
  }

  Points3 v_;
  Faces f_;
  Mat3 aniso_;
  Mat3 aniso_inv_;
  Index nv_;
  Index nf_;
  std::vector<std::vector<Index>> vf_;
  std::vector<Cluster> clusters_;
  std::vector<Index> rep_;  // vertex -> its registered cluster's representative, else itself
  Points3 pos_;             // vertex positions with registered clusters merged
  Mesh result_;
};

}  // namespace polatory::isosurface
