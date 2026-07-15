#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <vector>

#include "disjoint_sets.hpp"
#include "quadric_position.hpp"
#include "utility.hpp"

namespace polatory::isosurface {

// Clusters the RMT surface as a standalone mesh step, so it can be re-run between smoothing passes.
// Each lattice node's vertices are split into connected components, and every component is merged
// into one vertex (a quotient). Forcing the merge can fold the surface onto itself, which shows up
// as a coincident opposite-winding face pair; dropping that pair leaves the merged vertex manifold.
// A detect-and-uncluster loop then forbids any remaining cluster whose merged vertex a defect
// finder still flags, until the result is manifold.
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
        resolution_(lattice.resolution()),
        nv_(v_.rows()),
        nf_(f_.rows()),
        vf_(nv_),
        cluster_of_(nv_, -1) {
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
    DisjointSets sets(nv_);
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

      Index rep = *std::min_element(component.begin(), component.end());
      Point3 position = clustered_position(component, lattice, vertex_node.at(rep));
      clusters_.emplace_back(std::move(component), rep, position);
      register_cluster(clusters_.size() - 1);
    }

    result_ = cluster();
  }

  Mesh result() && { return std::move(result_); }

 private:
  Mesh cluster() {
    while (true) {
      auto [mesh, vertex_ci] = clustered_mesh();
      mesh = remove_back_to_back(mesh);
      MeshDefectsFinder defects(mesh, resolution_);
      auto vis = defects.singular_vertices();
      auto fis = defects.intersecting_faces();
      boost::unordered_flat_set<Index> flagged(vis.begin(), vis.end());
      for (auto fi : fis) {
        for (auto k = 0; k < 3; k++) {
          flagged.insert(mesh.faces()(fi, k));
        }
      }

      // Each vertex maps to the clusters whose merged vertex shares a face with it.
      boost::unordered_flat_map<Index, boost::unordered_flat_set<std::size_t>> incident;
      for (auto f : mesh.faces().rowwise()) {
        for (auto k = 0; k < 3; k++) {
          auto it = vertex_ci.find(f(k));
          if (it != vertex_ci.end()) {
            incident[f((k + 1) % 3)].insert(it->second);
            incident[f((k + 2) % 3)].insert(it->second);
          }
        }
      }

      bool changed = false;
      auto rollback = [&](std::size_t ci) {
        auto& cluster = clusters_.at(ci);
        if (!cluster.deleted) {
          unregister_cluster(ci);
          cluster.deleted = true;
          changed = true;
        }
      };

      // Undo the merge behind each flagged vertex: its own if it is a merged vertex, otherwise
      // every merge incident to it, since which neighbor spoiled it is unknown.
      for (auto v : flagged) {
        if (auto it = vertex_ci.find(v); it != vertex_ci.end()) {
          rollback(it->second);
        } else if (auto jt = incident.find(v); jt != incident.end()) {
          for (auto ci : jt->second) {
            rollback(ci);
          }
        }
      }

      if (!changed) {
        return mesh;
      }
    }
  }

  // Second return maps each merged output vertex to the cluster it came from.
  std::pair<Mesh, boost::unordered_flat_map<Index, std::size_t>> clustered_mesh() const {
    Points3 vertices(nv_, 3);
    Faces faces(nf_, 3);
    Index nv = 0;
    Index nf = 0;
    std::vector<Index> vv(nv_, -1);
    for (auto f : f_.rowwise()) {
      Face g;
      for (auto k = 0; k < 3; k++) {
        auto ci = cluster_of_.at(f(k));
        auto v = ci < 0 ? f(k) : clusters_.at(ci).rep;
        if (vv.at(v) < 0) {
          vv.at(v) = nv;
          vertices.row(nv) = ci < 0 ? v_.row(f(k)) : clusters_.at(ci).position;
          nv++;
        }
        g(k) = vv.at(v);
      }
      if (g(0) == g(1) || g(1) == g(2) || g(2) == g(0)) {
        continue;  // the cluster collapsed this face to fewer than three vertices
      }
      faces.row(nf) = g;
      nf++;
    }
    vertices.conservativeResize(nv, Eigen::NoChange);
    faces.conservativeResize(nf, Eigen::NoChange);

    boost::unordered_flat_map<Index, std::size_t> vertex_ci;
    for (std::size_t ci = 0; ci < clusters_.size(); ci++) {
      const auto& cluster = clusters_.at(ci);
      if (cluster.deleted) {
        continue;
      }
      vertex_ci[vv.at(cluster.rep)] = ci;
    }

    return {Mesh{std::move(vertices), std::move(faces)}, vertex_ci};
  }

  // Merges the cluster's vertices to the quadric minimizer over its incident face planes,
  // keeping a crease/corner instead of averaging it away.
  Point3 clustered_position(const std::vector<Index>& cluster, const PrimitiveLattice& lattice,
                            const LatticeCoordinates& node) const {
    boost::unordered_flat_set<Index> fis;
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
    auto p = quadric_position(v_(cluster, kAll), triangles, aniso_, aniso_inv_, lattice, node);
    return snap_to_bbox(p, lattice.bbox(), lattice.resolution());
  }

  void register_cluster(std::size_t ci) {
    for (auto v : clusters_.at(ci).vertices) {
      cluster_of_.at(v) = static_cast<Index>(ci);
    }
  }

  // Drops every coincident opposite-winding face pair -- a zero-volume fold left where a forced
  // merge folded the surface onto itself. The pair adds two faces to each of its three edges, which
  // the drop removes. Whether the drop is safe depends on the edge's face count beforehand:
  //   - Two: the pair alone. The drop leaves zero, so the edge vanishes. Safe.
  //   - Three: the drop leaves one face, a boundary edge -- a hole. Kept.
  //   - Four: the pair plus a sheet it folded onto. The drop leaves the sheet's two. Safe.
  //   - Five or more: the drop leaves three or more, still non-manifold. Kept.
  static Mesh remove_back_to_back(const Mesh& mesh) {
    const auto& f = mesh.faces();
    boost::unordered_flat_map<Edge, int, EdgeHash> edge_faces;
    for (Index fi = 0; fi < f.rows(); fi++) {
      for (auto k = 0; k < 3; k++) {
        edge_faces[{f(fi, k), f(fi, (k + 1) % 3)}]++;
      }
    }
    boost::unordered_flat_map<std::array<Index, 3>, std::vector<Index>,
                              boost::hash<std::array<Index, 3>>>
        by_vertices;
    for (Index fi = 0; fi < f.rows(); fi++) {
      std::array<Index, 3> key{f(fi, 0), f(fi, 1), f(fi, 2)};
      std::sort(key.begin(), key.end());
      by_vertices[key].push_back(fi);
    }
    std::vector<bool> dropped(f.rows(), false);
    for (const auto& [key, fis] : by_vertices) {
      if (fis.size() != 2 || same_winding(f, fis.at(0), fis.at(1))) {
        continue;
      }
      auto fi = fis.at(0);
      auto safe = true;
      for (auto k = 0; k < 3; k++) {
        auto count = edge_faces.at(Edge{f(fi, k), f(fi, (k + 1) % 3)});
        safe = safe && (count == 2 || count == 4);
      }
      if (safe) {
        dropped.at(fis.at(0)) = true;
        dropped.at(fis.at(1)) = true;
      }
    }
    Index out_nf = 0;
    for (Index fi = 0; fi < f.rows(); fi++) {
      if (!dropped.at(fi)) {
        out_nf++;
      }
    }
    Faces out(out_nf, 3);
    Index r = 0;
    for (Index fi = 0; fi < f.rows(); fi++) {
      if (!dropped.at(fi)) {
        out.row(r++) = f.row(fi);
      }
    }
    return {mesh.vertices(), std::move(out)};
  }

  static bool same_winding(const Faces& f, Index a, Index b) {
    for (auto s = 0; s < 3; s++) {
      if (f(a, 0) == f(b, s) && f(a, 1) == f(b, (s + 1) % 3) && f(a, 2) == f(b, (s + 2) % 3)) {
        return true;
      }
    }
    return false;
  }

  void unregister_cluster(std::size_t ci) {
    for (auto v : clusters_.at(ci).vertices) {
      cluster_of_.at(v) = -1;
    }
  }

  Points3 v_;
  Faces f_;
  Mat3 aniso_;
  Mat3 aniso_inv_;
  double resolution_;
  Index nv_;
  Index nf_;
  std::vector<std::vector<Index>> vf_;
  std::vector<Cluster> clusters_;
  std::vector<Index> cluster_of_;  // vertex -> its registered cluster, or -1 if unclustered
  Mesh result_;
};

}  // namespace polatory::isosurface
