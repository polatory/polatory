#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/predicates.hpp>
#include <unordered_map>

#include "dense_undirected_graph.hpp"
#include "indexer.hpp"

namespace polatory::isosurface {

MeshDefectsFinder::MeshDefectsFinder(const Mesh& mesh)
    : vertices_(mesh.vertices()), faces_(mesh.faces()), vf_map_(vertices_.rows()) {
  auto n_faces = faces_.rows();
  for (Index fi = 0; fi < n_faces; fi++) {
    auto f = faces_.row(fi);
    vf_map_.at(f(0)).push_back(fi);
    vf_map_.at(f(1)).push_back(fi);
    vf_map_.at(f(2)).push_back(fi);
  }
}

// Currently, only intersections between faces that share a single vertex are checked.
std::vector<Index> MeshDefectsFinder::intersecting_faces() const {
  std::vector<Index> result;

  auto n_vertices = vertices_.rows();
#pragma omp parallel
  {
    std::vector<Index> local_result;

#pragma omp for schedule(guided)
    for (Index vi = 0; vi < n_vertices; vi++) {
      const auto& fis = vf_map_.at(vi);

      auto n_faces = static_cast<Index>(fis.size());
      for (Index i = 0; i < n_faces - 1; i++) {
        auto fi = fis.at(i);
        auto a = next_vertex(fi, vi);
        auto b = prev_vertex(fi, vi);
        for (Index j = i + 1; j < n_faces; j++) {
          auto fj = fis.at(j);
          auto c = next_vertex(fj, vi);
          auto d = prev_vertex(fj, vi);

          if (b == c || a == d || a == c || b == d) {
            // Skip pairs of adjacent faces.
            // The last two conditions are included for handling faces around non-manifold edges.
            continue;
          }

          if (edge_face_intersect({a, b}, fj) || edge_face_intersect({c, d}, fi)) {
            local_result.push_back(fi);
            local_result.push_back(fj);
          }
        }
      }
    }

#pragma omp critical
    result.insert(result.end(), local_result.begin(), local_result.end());
  }

  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());

  return result;
}

std::vector<Index> MeshDefectsFinder::singular_vertices() const {
  std::vector<Index> result;

  auto n_vertices = vertices_.rows();
#pragma omp parallel
  {
    std::vector<Index> local_result;

#pragma omp for schedule(guided)
    for (Index vi = 0; vi < n_vertices; vi++) {
      const auto& fis = vf_map_.at(vi);

      if (fis.empty()) {
        // An isolated vertex.
        continue;
      }

      std::unordered_map<Index, Index> to_local_vi;
      for (auto fi : fis) {
        to_local_vi.emplace(next_vertex(fi, vi), to_local_vi.size());
        to_local_vi.emplace(prev_vertex(fi, vi), to_local_vi.size());
      }

      auto order = static_cast<Index>(to_local_vi.size());

      // The graph that represents the link complex of the vertex.
      DenseUndirectedGraph g(IdentityIndexer{order});

      for (auto fi : fis) {
        auto i = to_local_vi.at(next_vertex(fi, vi));
        auto j = to_local_vi.at(prev_vertex(fi, vi));
        g.add_edge(i, j);
      }

      // Check if the graph is a cycle or a path (in case of a boundary vertex).
      // NOLINTNEXTLINE(readability-simplify-boolean-expr)
      if (!(g.is_simple() && g.is_connected() && g.max_degree() <= 2)) {
        local_result.push_back(vi);
      }
    }

#pragma omp critical
    result.insert(result.end(), local_result.begin(), local_result.end());
  }

  return result;
}

Index MeshDefectsFinder::next_vertex(Index fi, Index vi) const {
  auto f = faces_.row(fi);
  if (f(0) == vi) {
    return f(1);
  }
  if (f(1) == vi) {
    return f(2);
  }
  return f(0);
}

Index MeshDefectsFinder::prev_vertex(Index fi, Index vi) const {
  auto f = faces_.row(fi);
  if (f(0) == vi) {
    return f(2);
  }
  if (f(1) == vi) {
    return f(0);
  }
  return f(1);
}

bool MeshDefectsFinder::edge_face_intersect(const Edge& e, Index fi) const {
  auto f = faces_.row(fi);

  geometry::Point3 p = vertices_.row(e.a);
  geometry::Point3 q = vertices_.row(e.b);
  geometry::Point3 a = vertices_.row(f(0));
  geometry::Point3 b = vertices_.row(f(1));
  geometry::Point3 c = vertices_.row(f(2));

  return segment3_triangle3_intersect(p, q, a, b, c);
}

}  // namespace polatory::isosurface
