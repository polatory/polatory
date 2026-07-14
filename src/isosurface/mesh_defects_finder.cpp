#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <boost/unordered/unordered_flat_map.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/predicates.hpp>

#include "dense_undirected_graph.hpp"

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

  auto folded = [this](Index a, Index b, Index c, Index d) {
    geometry::Point3 pa = vertices_.row(a);
    geometry::Point3 pb = vertices_.row(b);
    geometry::Point3 pc = vertices_.row(c);
    geometry::Point3 pd = vertices_.row(d);

    return isosurface::folded(pa, pb, pc, pd);
  };

  auto edge_face_intersect = [this](Index a, Index b, Index p, Index q, Index r) {
    geometry::Point3 pa = vertices_.row(a);
    geometry::Point3 pb = vertices_.row(b);
    geometry::Point3 pp = vertices_.row(p);
    geometry::Point3 pq = vertices_.row(q);
    geometry::Point3 pr = vertices_.row(r);

    return segment3_triangle3_intersect(pa, pb, pp, pq, pr);
  };

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
        auto a = next_vertex(fi, v);
        auto b = prev_vertex(fi, v);
        for (Index j = i + 1; j < n_faces; j++) {
          auto fj = fis.at(j);
          auto c = next_vertex(fj, v);
          auto d = prev_vertex(fj, v);

          if (b == c) {
            if (folded(v, b /* = c */, a, d)) {
              local_result.push_back(fi);
              local_result.push_back(fj);
            }
            continue;
          }

          if (a == d) {
            if (folded(v, a /* = d */, b, c)) {
              local_result.push_back(fi);
              local_result.push_back(fj);
            }
            continue;
          }

          if (a == c || b == d) {
            // Inconsistently oriented faces.
            continue;
          }

          if (edge_face_intersect(a, b, v, c, d) || edge_face_intersect(c, d, v, a, b)) {
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
