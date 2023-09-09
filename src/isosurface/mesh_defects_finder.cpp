#include <polatory/isosurface/dense_undirected_graph.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <unordered_map>

namespace polatory::isosurface {

mesh_defects_finder::mesh_defects_finder(const vertices_type& vertices, const faces_type& faces)
    : vertices_(vertices), faces_(faces), vf_map_(vertices_.rows()) {
  auto n_faces = faces.rows();
  for (index_t fi = 0; fi < n_faces; fi++) {
    auto f = faces.row(fi);
    vf_map_.at(f(0)).push_back(fi);
    vf_map_.at(f(1)).push_back(fi);
    vf_map_.at(f(2)).push_back(fi);
  }
}

// Currently, only intersections between faces that share a single vertex are checked.
std::unordered_set<index_t> mesh_defects_finder::intersecting_faces() const {
  std::unordered_set<index_t> result;

  auto n_vertices = vertices_.rows();
#pragma omp parallel for schedule(guided, 32)
  for (index_t vi = 0; vi < n_vertices; vi++) {
    const auto& fis = vf_map_.at(vi);

    auto n_faces = static_cast<index_t>(fis.size());
    for (index_t i = 0; i < n_faces - 1; i++) {
      auto fi = fis.at(i);
      auto a = next_vertex(fi, vi);
      auto b = prev_vertex(fi, vi);
      for (index_t j = i + 1; j < n_faces; j++) {
        auto fj = fis.at(j);
        auto c = next_vertex(fj, vi);
        auto d = prev_vertex(fj, vi);

        if (b == c || a == d) {
          // Skip the pair of adjacent faces.
          // As faces are oriented, we don't need to check other combinations.
          continue;
        }

        if (segment_triangle_intersect(a, b, fj) || segment_triangle_intersect(c, d, fi)) {
#pragma omp critical
          {
            result.insert(fi);
            result.insert(fj);
          }
        }
      }
    }
  }

  return result;
}

std::unordered_set<index_t> mesh_defects_finder::singular_vertices() const {
  std::unordered_set<index_t> result;

  auto n_vertices = vertices_.rows();
#pragma omp parallel for schedule(guided, 32)
  for (index_t vi = 0; vi < n_vertices; vi++) {
    const auto& fis = vf_map_.at(vi);

    if (fis.empty()) {
      // An isolated vertex.
      continue;
    }

    std::unordered_map<index_t, index_t> to_local_vi;
    for (auto fi : fis) {
      to_local_vi.emplace(next_vertex(fi, vi), to_local_vi.size());
      to_local_vi.emplace(prev_vertex(fi, vi), to_local_vi.size());
    }

    auto order = static_cast<index_t>(to_local_vi.size());

    // The graph that represents the link complex of the vertex.
    dense_undirected_graph g(order);

    for (auto fi : fis) {
      auto i = to_local_vi.at(next_vertex(fi, vi));
      auto j = to_local_vi.at(prev_vertex(fi, vi));
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

bool mesh_defects_finder::line_triangle_intersect(index_t vi, index_t vj, index_t fi) const {
  const auto f = faces_.row(fi);

  const auto e1 = vertices_.row(f(1)) - vertices_.row(f(0));
  const auto e2 = vertices_.row(f(2)) - vertices_.row(f(0));

  const auto dir = vertices_.row(vj) - vertices_.row(vi);
  const auto p = dir.cross(e2);
  // det = [e1, dir, e2] (scalar triple product of dir, e2 and e1)
  const auto inv_det = 1.0 / p.dot(e1);
  const auto s = vertices_.row(vi) - vertices_.row(f(0));
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

bool mesh_defects_finder::segment_plane_intersect(index_t vi, index_t vj, index_t fi) const {
  const auto f = faces_.row(fi);

  const auto e1 = vertices_.row(f(1)) - vertices_.row(f(0));
  const auto e2 = vertices_.row(f(2)) - vertices_.row(f(0));

  const auto n = e1.cross(e2);
  const auto sign1 = n.dot(vertices_.row(vi) - vertices_.row(f(0)));
  const auto sign2 = n.dot(vertices_.row(vj) - vertices_.row(f(0)));
  return sign1 * sign2 < 0.0;
}

bool mesh_defects_finder::segment_triangle_intersect(index_t vi, index_t vj, index_t fi) const {
  return segment_plane_intersect(vi, vj, fi) && line_triangle_intersect(vi, vj, fi);
}

index_t mesh_defects_finder::next_vertex(index_t fi, index_t vi) const {
  auto f = faces_.row(fi);
  if (f(0) == vi) {
    return f(1);
  }
  if (f(1) == vi) {
    return f(2);
  }
  return f(0);
}

index_t mesh_defects_finder::prev_vertex(index_t fi, index_t vi) const {
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
