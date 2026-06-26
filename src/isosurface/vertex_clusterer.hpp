#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
  using Vector3 = geometry::Vector3;

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

    // Union-find forest grouping each node's vertices into connected components over same-node
    // edges.
    std::vector<Index> parent(nv_);
    std::iota(parent.begin(), parent.end(), Index{0});
    auto find = [&](Index v) {
      while (parent.at(v) != v) {
        parent.at(v) = parent.at(parent.at(v));
        v = parent.at(v);
      }
      return v;
    };
    for (auto f : f_.rowwise()) {
      for (auto k = 0; k < 3; k++) {
        Index v0 = f(k);
        Index v1 = f((k + 1) % 3);
        if (vertex_node.at(v0) == vertex_node.at(v1)) {
          auto r0 = find(v0);
          auto r1 = find(v1);
          if (r0 != r1) {
            parent.at(r0) = r1;
          }
        }
      }
    }

    std::unordered_map<Index, std::vector<Index>> components;
    for (Index v = 0; v < nv_; v++) {
      components[find(v)].push_back(v);
    }

    rep_.resize(nv_);
    std::iota(rep_.begin(), rep_.end(), Index{0});
    pos_ = v_;

    const auto& lo = lattice.first_extended_bbox().min();
    const auto& hi = lattice.first_extended_bbox().max();
    for (auto& [_, component] : components) {
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
        Point3 position = clustered_position(component);
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
    std::unordered_map<Index, std::vector<Index>> link;
    Index n_edges = 0;
    for (auto fi : faces) {
      std::array<Index, 3> out{};
      auto n_out = 0;
      for (auto k = 0; k < 3; k++) {
        if (!in_component.contains(f_(fi, k))) {
          out.at(n_out++) = f_(fi, k);
        }
      }
      if (n_out == 2) {
        link[out.at(0)].push_back(out.at(1));
        link[out.at(1)].push_back(out.at(0));
        n_edges++;
      }
    }
    if (n_edges < 3 || static_cast<Index>(link.size()) != n_edges) {
      return false;
    }
    for (const auto& [w, nbrs] : link) {
      if (nbrs.size() != 2) {
        return false;
      }
    }
    Index start = link.begin()->first;
    Index prev = -1;
    Index cur = start;
    Index seen = 0;
    while (true) {
      seen++;
      const auto& nbrs = link.at(cur);
      Index next = nbrs.at(0) == prev ? nbrs.at(1) : nbrs.at(0);
      prev = cur;
      cur = next;
      if (cur == start) {
        break;
      }
      if (seen > n_edges) {
        return false;
      }
    }
    return seen == n_edges;
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

  // The merged vertex minimizes the area-weighted squared distance to the cluster's incident face
  // planes:  x = argmin_x  sum_f  area_f (n_f.x + d_f)^2  -- keeping a crease/corner instead of
  // averaging it away. Directions the planes leave free (a flat patch) keep the centroid; x is
  // clamped to the cluster's AABB. Measured in the aniso-transformed frame so the fit respects the
  // anisotropic resolution.
  Point3 clustered_position(const std::vector<Index>& cluster) const {
    Points3 a_points = geometry::transform_points<3>(aniso_, v_(cluster, kAll));
    Point3 centroid = a_points.colwise().mean();
    Point3 lo = a_points.colwise().minCoeff();
    Point3 hi = a_points.colwise().maxCoeff();

    std::unordered_set<Index> fis;
    for (auto v : cluster) {
      for (auto fi : vf_.at(v)) {
        fis.insert(fi);
      }
    }

    // Accumulate aa and bb, the matrix and vector of that energy's normal equation.
    Mat3 aa = Mat3::Zero();
    Vector3 bb = Vector3::Zero();
    for (auto fi : fis) {
      Point3 p0 = geometry::transform_point<3>(aniso_, v_.row(f_(fi, 0)));
      Point3 p1 = geometry::transform_point<3>(aniso_, v_.row(f_(fi, 1)));
      Point3 p2 = geometry::transform_point<3>(aniso_, v_.row(f_(fi, 2)));
      Vector3 n = (p1 - p0).cross(p2 - p0);
      auto w = n.norm();
      if (w == 0.0) {
        continue;
      }
      n /= w;
      auto d = -n.dot(p0);
      aa += w * n.transpose() * n;
      bb += w * d * n;
    }

    // Solve the normal equation aa x = -bb in a rank-revealing manner: move off the centroid only
    // when aa constrains more than one direction -- a crease (rank 2) or corner (rank 3) -- and
    // along just those. A flat patch (rank 1) stays at the centroid.
    Eigen::SelfAdjointEigenSolver<Mat3> es(aa);
    auto floor = 1e-3 * es.eigenvalues()(2);
    Vector3 y = Vector3::Zero();
    if (es.eigenvalues()(1) > floor) {
      Vector3 r = -(centroid * aa + bb);
      for (auto k = 0; k < 3; k++) {
        auto eval = es.eigenvalues()(k);
        if (eval > floor) {
          Vector3 evec = es.eigenvectors().col(k).transpose();
          y += (r.dot(evec) / eval) * evec;
        }
      }
    }
    Point3 x = (centroid + y).cwiseMax(lo).cwiseMin(hi);
    return geometry::transform_point<3>(aniso_inv_, x);
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
