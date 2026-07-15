#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <boost/unordered/unordered_flat_map.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>

#include "dense_undirected_graph.hpp"
#include "utility.hpp"

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

bool MeshDefectsFinder::intersect(Index fi, Index fj) const {
  auto a = faces_.row(fi);
  auto b = faces_.row(fj);
  return triangles_intersect(vertices_.row(a(0)), vertices_.row(a(1)), vertices_.row(a(2)),
                             vertices_.row(b(0)), vertices_.row(b(1)), vertices_.row(b(2)));
}

// Only pairs of faces that share a vertex are checked.
std::vector<Index> MeshDefectsFinder::intersecting_faces() const {
  std::vector<Index> result;

  auto n_vertices = vertices_.rows();
#pragma omp parallel
  {
    std::vector<Index> local_result;

#pragma omp for schedule(guided)
    for (Index v = 0; v < n_vertices; v++) {
      const auto& fis = vf_map_.at(v);

      auto n_faces = static_cast<Index>(fis.size());
      for (Index i = 0; i < n_faces - 1; i++) {
        auto fi = fis.at(i);
        for (Index j = i + 1; j < n_faces; j++) {
          auto fj = fis.at(j);
          if (intersect(fi, fj)) {
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

      boost::unordered_flat_map<Index, Index> to_local_vi;
      for (auto fi : fis) {
        to_local_vi.emplace(next_vertex(fi, vi), to_local_vi.size());
        to_local_vi.emplace(prev_vertex(fi, vi), to_local_vi.size());
      }

      auto order = static_cast<Index>(to_local_vi.size());

      // The graph that represents the link complex of the vertex.
      DenseUndirectedGraph g(order);

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

}  // namespace polatory::isosurface
