#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <polatory/isosurface/dense_undirected_graph.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <unordered_map>

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

#pragma omp for schedule(guided, 32)
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

          if (edge_face_intersect(a, b, fj) || edge_face_intersect(c, d, fi)) {
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

#pragma omp for schedule(guided, 32)
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

double orient2d_inexact(const geometry::Point2& a, const geometry::Point2& b,
                        const geometry::Point2& c) {
  Mat2 m;
  m << a(0) - c(0), a(1) - c(1), b(0) - c(0), b(1) - c(1);
  return m.determinant();
}

double orient3d_inexact(const geometry::Point3& a, const geometry::Point3& b,
                        const geometry::Point3& c, const geometry::Point3& d) {
  Mat3 m;
  m << a(0) - d(0), a(1) - d(1), a(2) - d(2), b(0) - d(0), b(1) - d(1), b(2) - d(2), c(0) - d(0),
      c(1) - d(1), c(2) - d(2);
  return m.determinant();
}

bool segment3_triangle3_intersect_coplanar(const geometry::Point3& p, const geometry::Point3& q,
                                           const geometry::Point3& a, const geometry::Point3& b,
                                           const geometry::Point3& c) {
  geometry::Vector3 n = (b - a).cross(c - a);
  auto abs_nx = std::abs(n(0));
  auto abs_ny = std::abs(n(1));
  auto abs_nz = std::abs(n(2));

  auto i = -1;
  auto j = -1;
  if (abs_nx >= abs_ny && abs_nx >= abs_nz) {
    std::tie(i, j) = n(0) > 0 ? std::make_tuple(1, 2) : std::make_tuple(2, 1);
  } else if (abs_ny >= abs_nx && abs_ny >= abs_nz) {
    std::tie(i, j) = n(1) > 0 ? std::make_tuple(2, 0) : std::make_tuple(0, 2);
  } else {
    std::tie(i, j) = n(2) > 0 ? std::make_tuple(0, 1) : std::make_tuple(1, 0);
  }

  geometry::Point2 p2(p(i), p(j));
  geometry::Point2 q2(q(i), q(j));
  geometry::Point2 a2(a(i), a(j));
  geometry::Point2 b2(b(i), b(j));
  geometry::Point2 c2(c(i), c(j));

  auto pqa = orient2d_inexact(p2, q2, a2);
  auto pqb = orient2d_inexact(p2, q2, b2);
  auto pqc = orient2d_inexact(p2, q2, c2);

  auto sign = [](double x) -> int { return x > 0.0 ? 1 : x < 0.0 ? -1 : 0; };
  auto make_class = [](int a, int b, int c) constexpr -> int {
    return 9 * (a + 1) + 3 * (b + 1) + c + 1;
  };
  switch (make_class(sign(pqa), sign(pqb), sign(pqc))) {
    case make_class(1, 1, 1):
      return false;

    case make_class(1, 1, -1):
    case make_class(0, 1, -1):
    case make_class(1, 0, -1):
    case make_class(0, 0, -1):
    case make_class(1, 1, 0):
      //    B   A                  A              B                                     B   A
      // P ------- Q   or   P -B----- Q   or   P -----A- Q   or   P -B---A- Q   or   P ---C--- Q
      //      C                    C              C                    C
      return orient2d_inexact(p2, c2, a2) >= 0.0 && orient2d_inexact(q2, b2, c2) >= 0.0;

    case make_class(1, -1, 1):
    case make_class(0, -1, 1):
    case make_class(1, -1, 0):
    case make_class(0, -1, 0):
    case make_class(1, 0, 1):
      //    A   C                  C              A                                     A   C
      // P ------- Q   or   P -A----- Q   or   P -----C- Q   or   P -A---C- Q   or   P ---B--- Q
      //      B                    B              B                    B
      return orient2d_inexact(p2, b2, c2) >= 0.0 && orient2d_inexact(q2, a2, b2) >= 0.0;

    case make_class(-1, 1, 1):
    case make_class(-1, 1, 0):
    case make_class(-1, 0, 1):
    case make_class(-1, 0, 0):
    case make_class(0, 1, 1):
      //    C   B                  B              C                                     C   B
      // P ------- Q   or   P -C----- Q   or   P -----B- Q   or   P -C---B- Q   or   P ---A--- Q
      //      A                    A              A                    A
      return orient2d_inexact(p2, a2, b2) >= 0.0 && orient2d_inexact(q2, c2, a2) >= 0.0;

    case make_class(1, -1, -1):
    // case make_class(1, 0, -1):
    // case make_class(1, -1, 0):
    case make_class(1, 0, 0):
    case make_class(0, -1, -1):
      //      A                    A              A                    A
      // P ------- Q   or   P -B----- Q   or   P -----C- Q   or   P -B---C- Q   or   P ---A--- Q
      //    B   C                  C              B                                     B   C
      return orient2d_inexact(p2, c2, a2) >= 0.0 && orient2d_inexact(q2, a2, b2) >= 0.0;

    case make_class(-1, 1, -1):
    // case make_class(-1, 1, 0):
    // case make_class(0, 1, -1):
    case make_class(0, 1, 0):
    case make_class(-1, 0, -1):
      //      B                    B              B                    B
      // P ------- Q   or   P -C----- Q   or   P -----A- Q   or   P -C---A- Q   or   P ---B--- Q
      //    C   A                  A              C                                     C   A
      return orient2d_inexact(p2, a2, b2) >= 0.0 && orient2d_inexact(q2, b2, c2) >= 0.0;

    case make_class(-1, -1, 1):
    // case make_class(0, -1, 1):
    // case make_class(-1, 0, 1):
    case make_class(0, 0, 1):
    case make_class(-1, -1, 0):
      //      C                    C              C                    C
      // P ------- Q   or   P -A----- Q   or   P -----B- Q   or   P -A---B- Q   or   P ---C--- Q
      //    A   B                  B              A                                     A   B
      return orient2d_inexact(p2, b2, c2) >= 0.0 && orient2d_inexact(q2, c2, a2) >= 0.0;

    case make_class(-1, -1, -1):
      return false;

    default:
      // The segment or the triangle is degenerate.
      // This can happen when the segment and the triangle are not coplanar.
      // Return true to enable further checks.
      return true;
  }
}

bool segment3_triangle3_intersect(const geometry::Point3& p, const geometry::Point3& q,
                                  const geometry::Point3& a, const geometry::Point3& b,
                                  const geometry::Point3& c) {
  auto abcp = orient3d_inexact(a, b, c, p);
  auto abcq = orient3d_inexact(a, b, c, q);

  if ((abcp > 0.0 && abcq > 0.0) || (abcp < 0.0 && abcq < 0.0)) {
    return false;
  }

  // For robustness.
  if (!segment3_triangle3_intersect_coplanar(p, q, a, b, c)) {
    return false;
  }

  if (abcp == 0.0 && abcq == 0.0) {
    return true;
  }

  auto pqab = orient3d_inexact(p, q, a, b);
  auto pqbc = orient3d_inexact(p, q, b, c);
  auto pqca = orient3d_inexact(p, q, c, a);

  return (pqab >= 0.0 && pqbc >= 0.0 && pqca >= 0.0) || (pqab <= 0.0 && pqbc <= 0.0 && pqca <= 0.0);
}

bool MeshDefectsFinder::edge_face_intersect(Index vi, Index vj, Index fi) const {
  auto f = faces_.row(fi);

  geometry::Point3 p = vertices_.row(vi);
  geometry::Point3 q = vertices_.row(vj);
  geometry::Point3 a = vertices_.row(f(0));
  geometry::Point3 b = vertices_.row(f(1));
  geometry::Point3 c = vertices_.row(f(2));

  return segment3_triangle3_intersect(p, q, a, b, c);
}

}  // namespace polatory::isosurface
