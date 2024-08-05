#include <Eigen/Geometry>
#include <Eigen/LU>
#include <polatory/isosurface/dense_undirected_graph.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <unordered_map>

namespace polatory::isosurface {

mesh_defects_finder::mesh_defects_finder(const Points& vertices, const Faces& faces)
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

        if (edge_face_intersect(a, b, fj) || edge_face_intersect(c, d, fi)) {
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

double orient2d_inexact(const geometry::point2d& a, const geometry::point2d& b,
                        const geometry::point2d& c) {
  geometry::matrix2d m;
  m << a(0) - c(0), a(1) - c(1), b(0) - c(0), b(1) - c(1);
  return m.determinant();
}

double orient3d_inexact(const geometry::point3d& a, const geometry::point3d& b,
                        const geometry::point3d& c, const geometry::point3d& d) {
  geometry::matrix3d m;
  m << a(0) - d(0), a(1) - d(1), a(2) - d(2), b(0) - d(0), b(1) - d(1), b(2) - d(2), c(0) - d(0),
      c(1) - d(1), c(2) - d(2);
  return m.determinant();
}

bool segment2_segment2_intersect(const geometry::point2d& p, const geometry::point2d& q,
                                 const geometry::point2d& r, const geometry::point2d& s) {
  auto pqr = orient2d_inexact(p, q, r);
  auto pqs = orient2d_inexact(p, q, s);
  auto rsp = orient2d_inexact(r, s, p);
  auto rsq = orient2d_inexact(r, s, q);

  return pqr * pqs <= 0.0 && rsp * rsq <= 0.0;
}

bool point2_triangle2_intersect(const geometry::point2d& p, const geometry::point2d& a,
                                const geometry::point2d& b, const geometry::point2d& c) {
  auto pab = orient2d_inexact(p, a, b);
  auto pbc = orient2d_inexact(p, b, c);
  auto pca = orient2d_inexact(p, c, a);

  return (pab >= 0.0 && pbc >= 0.0 && pca >= 0.0) || (pab <= 0.0 && pbc <= 0.0 && pca <= 0.0);
}

bool segment2_triangle2_intersect(const geometry::point2d& p, const geometry::point2d& q,
                                  const geometry::point2d& a, const geometry::point2d& b,
                                  const geometry::point2d& c) {
  return segment2_segment2_intersect(p, q, a, b) || segment2_segment2_intersect(p, q, b, c) ||
         segment2_segment2_intersect(p, q, c, a) || point2_triangle2_intersect(p, a, b, c) ||
         // This check is redundant, though.
         point2_triangle2_intersect(q, a, b, c);
}

bool segment3_triangle3_intersect_coplanar(const geometry::point3d& p, const geometry::point3d& q,
                                           const geometry::point3d& a, const geometry::point3d& b,
                                           const geometry::point3d& c) {
  geometry::vector3d n = (b - a).cross(c - a);
  auto abs_nx = std::abs(n(0));
  auto abs_ny = std::abs(n(1));
  auto abs_nz = std::abs(n(2));

  if (abs_nx >= abs_ny && abs_nx >= abs_nz) {
    geometry::point2d p_yz(p(1), p(2));
    geometry::point2d q_yz(q(1), q(2));
    geometry::point2d a_yz(a(1), a(2));
    geometry::point2d b_yz(b(1), b(2));
    geometry::point2d c_yz(c(1), c(2));

    return segment2_triangle2_intersect(p_yz, q_yz, a_yz, b_yz, c_yz);
  }

  if (abs_ny >= abs_nx && abs_ny >= abs_nz) {
    geometry::point2d p_zx(p(2), p(0));
    geometry::point2d q_zx(q(2), q(0));
    geometry::point2d a_zx(a(2), a(0));
    geometry::point2d b_zx(b(2), b(0));
    geometry::point2d c_zx(c(2), c(0));

    return segment2_triangle2_intersect(p_zx, q_zx, a_zx, b_zx, c_zx);
  }

  geometry::point2d p_xy(p(0), p(1));
  geometry::point2d q_xy(q(0), q(1));
  geometry::point2d a_xy(a(0), a(1));
  geometry::point2d b_xy(b(0), b(1));
  geometry::point2d c_xy(c(0), c(1));

  return segment2_triangle2_intersect(p_xy, q_xy, a_xy, b_xy, c_xy);
}

bool segment3_triangle3_intersect(const geometry::point3d& p, const geometry::point3d& q,
                                  const geometry::point3d& a, const geometry::point3d& b,
                                  const geometry::point3d& c) {
  auto abcp = orient3d_inexact(a, b, c, p);
  auto abcq = orient3d_inexact(a, b, c, q);

  if (std::abs(abcp) < 1e-10) {
    abcp = 0.0;
  }
  if (std::abs(abcq) < 1e-10) {
    abcq = 0.0;
  }

  if ((abcp > 0.0 && abcq > 0.0) || (abcp < 0.0 && abcq < 0.0)) {
    return false;
  }

  if (abcp == 0.0 && abcq == 0.0) {
    return segment3_triangle3_intersect_coplanar(p, q, a, b, c);
  }

  auto pqab = orient3d_inexact(p, q, a, b);
  auto pqbc = orient3d_inexact(p, q, b, c);
  auto pqca = orient3d_inexact(p, q, c, a);

  return (pqab >= 0.0 && pqbc >= 0.0 && pqca >= 0.0) || (pqab <= 0.0 && pqbc <= 0.0 && pqca <= 0.0);
}

bool mesh_defects_finder::edge_face_intersect(index_t vi, index_t vj, index_t fi) const {
  auto f = faces_.row(fi);

  const auto& p = vertices_.row(vi);
  const auto& q = vertices_.row(vj);
  const auto& a = vertices_.row(f(0));
  const auto& b = vertices_.row(f(1));
  const auto& c = vertices_.row(f(2));

  return segment3_triangle3_intersect(p, q, a, b, c);
}

}  // namespace polatory::isosurface
