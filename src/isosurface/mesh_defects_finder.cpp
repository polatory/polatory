#include <map>
#include <polatory/isosurface/dense_undirected_graph.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>

namespace polatory::isosurface {

mesh_defects_finder::mesh_defects_finder(const std::vector<geometry::point3d>& vertices,
                                         const std::vector<face>& faces)
    : vertices_(vertices), vf_map_(vertices_.size()) {
  for (const auto& f : faces) {
    vf_map_.at(f[0]).push_back({f[0], f[1], f[2]});
    vf_map_.at(f[1]).push_back({f[1], f[2], f[0]});
    vf_map_.at(f[2]).push_back({f[2], f[0], f[1]});
  }
}

// Currently, intersections only between faces that share a single vertex are detected.
std::set<face> mesh_defects_finder::intersecting_faces() const {
  std::set<face> result;

  auto n_vertices = static_cast<vertex_index>(vertices_.size());
#pragma omp parallel for schedule(guided, 32)
  for (vertex_index vi = 0; vi < n_vertices; vi++) {
    const auto& faces = vf_map_.at(vi);

    auto n_faces = static_cast<int>(faces.size());
    for (auto i = 0; i < n_faces - 1; i++) {
      const auto& f1 = faces.at(i);
      for (auto j = i + 1; j < n_faces; j++) {
        const auto& f2 = faces.at(j);

        if (f1[1] == f2[2] || f1[2] == f2[1]) {
          // Skip the pair of adjacent faces.
          // As faces are oriented, we don't need to check other combinations.
          continue;
        }

        if (segment_triangle_intersect(f1[1], f1[2], f2) ||
            segment_triangle_intersect(f2[1], f2[2], f1)) {
#pragma omp critical
          {
            result.insert(f1);
            result.insert(f2);
          }
        }
      }
    }
  }

  return result;
}

std::set<vertex_index> mesh_defects_finder::singular_vertices() const {
  std::set<vertex_index> result;

  auto n_vertices = static_cast<vertex_index>(vertices_.size());
#pragma omp parallel for schedule(guided, 32)
  for (vertex_index vi = 0; vi < n_vertices; vi++) {
    const auto& faces = vf_map_.at(vi);

    if (faces.empty()) {
      // An isolated vertex.
      continue;
    }

    std::map<vertex_index, index_t> vi_to_index;
    for (const auto& f : faces) {
      vi_to_index.emplace(f[1], -1);
      vi_to_index.emplace(f[2], -1);
    }

    auto order = static_cast<index_t>(vi_to_index.size());

    std::vector<vertex_index> index_to_vi;
    index_to_vi.reserve(order);
    for (auto& vi_index : vi_to_index) {
      auto index = static_cast<index_t>(index_to_vi.size());
      index_to_vi.push_back(vi_index.first);
      vi_index.second = index;
    }

    // The graph that represents the link complex of the vertex.
    dense_undirected_graph g(order);

    for (const auto& f : faces) {
      auto i = vi_to_index.at(f[1]);
      auto j = vi_to_index.at(f[2]);
      g.add_edge(i, j);
    }

    // Check if the graph is a cycle or a path (in case of a boundary vertex).
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    if (!(g.is_simple() && g.is_connected() && g.max_degree() <= 2)) {
#pragma omp critical
      result.insert(vi);
    }
  }

  return result;
}

bool mesh_defects_finder::line_triangle_intersect(vertex_index s1, vertex_index s2,
                                                  const face& f) const {
  const auto e1 = vertices_.at(f[1]) - vertices_.at(f[0]);
  const auto e2 = vertices_.at(f[2]) - vertices_.at(f[0]);

  const auto dir = vertices_.at(s2) - vertices_.at(s1);
  const auto p = dir.cross(e2);
  // det = [e1, dir, e2] (scalar triple product of dir, e2 and e1)
  const auto inv_det = 1.0 / p.dot(e1);
  const auto s = vertices_.at(s1) - vertices_.at(f[0]);
  const auto u = inv_det * s.dot(p);
  if (u < 0.0 || u > 1.0) {
    return false;
  }
  const auto q = s.cross(e1);
  const auto v = inv_det * dir.dot(q);
  if (v < 0.0 || v > 1.0 || u + v > 1.0) {
    return false;
  }
  if (u + v > 1.0) {
    return false;
  }

  return true;
}

bool mesh_defects_finder::segment_plane_intersect(vertex_index s1, vertex_index s2,
                                                  const face& f) const {
  const auto e1 = vertices_.at(f[1]) - vertices_.at(f[0]);
  const auto e2 = vertices_.at(f[2]) - vertices_.at(f[0]);

  const auto n = e1.cross(e2);
  const auto sign1 = n.dot(vertices_.at(s1) - vertices_.at(f[0]));
  const auto sign2 = n.dot(vertices_.at(s2) - vertices_.at(f[0]));
  return sign1 * sign2 < 0.0;
}

bool mesh_defects_finder::segment_triangle_intersect(vertex_index s1, vertex_index s2,
                                                     const face& f) const {
  return segment_plane_intersect(s1, s2, f) && line_triangle_intersect(s1, s2, f);
}

}  // namespace polatory::isosurface
