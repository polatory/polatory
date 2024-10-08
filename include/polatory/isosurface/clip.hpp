#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <iterator>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface {

class MeshClipper {
  using Bbox = geometry::Bbox3;
  using Mat = Mat3;
  using Point = geometry::Point3;
  using Point2 = geometry::Point2;
  using Triangle = Mat3;
  using Vector = geometry::Vector3;

 public:
  MeshClipper(const Mesh& mesh, const Bbox& bbox) {
    const auto& vertices = mesh.vertices();
    const auto& faces = mesh.faces();

    std::vector<Triangle> triangles;
    for (auto f : faces.rowwise()) {
      Point p = vertices.row(f(0));
      Point q = vertices.row(f(1));
      Point r = vertices.row(f(2));
      triangles.push_back((Triangle() << p, q, r).finished());
    }

    std::array<Mat, 6> permutations{
        (Mat() << 1, 0, 0, 0, 1, 0, 0, 0, 1).finished(),   // x, y, z
        (Mat() << -1, 0, 0, 0, 0, 1, 0, 1, 0).finished(),  // -x, z, y
        (Mat() << 0, 1, 0, 0, 0, 1, 1, 0, 0).finished(),   // y, z, x
        (Mat() << 0, -1, 0, 1, 0, 0, 0, 0, 1).finished(),  // -y, x, z
        (Mat() << 0, 0, 1, 1, 0, 0, 0, 1, 0).finished(),   // z, x, y
        (Mat() << 0, 0, -1, 0, 1, 0, 1, 0, 0).finished()   // -z, y, x
    };
    std::array<double, 6> thresholds{bbox.max()(0),  -bbox.min()(0), bbox.max()(1),
                                     -bbox.min()(1), bbox.max()(2),  -bbox.min()(2)};

    std::vector<Triangle> clipped;
    for (auto face = 0; face < 6; face++) {
      const auto& perm = permutations.at(face);
      auto threshold = thresholds.at(face);
      for (auto& tri : triangles) {
        tri = geometry::transform_points<3>(perm, tri);
        clip(tri, threshold, clipped);
      }
      for (auto& tri : clipped) {
        tri = geometry::transform_points<3>(perm.transpose(), tri);
      }
      std::swap(triangles, clipped);
      clipped.clear();
    }

    std::unordered_map<Point, Index, PointHash> vertex_map;
    for (const auto& tri : triangles) {
      for (auto v : tri.rowwise()) {
        if (!vertex_map.contains(v)) {
          vertex_map.emplace(v, static_cast<Index>(vertex_map.size()));
        }
      }
    }

    geometry::Points3 clipped_vertices(static_cast<Index>(vertex_map.size()), 3);
    Faces clipped_faces(static_cast<Index>(triangles.size()), 3);

    for (const auto& pair : vertex_map) {
      clipped_vertices.row(pair.second) = pair.first;
    }

    for (std::size_t i = 0; i < triangles.size(); i++) {
      const auto& tri = triangles.at(i);
      auto face = clipped_faces.row(static_cast<Index>(i));
      face(0) = vertex_map.at(tri.row(0));
      face(1) = vertex_map.at(tri.row(1));
      face(2) = vertex_map.at(tri.row(2));
    }

    clipped_mesh_ = Mesh(std::move(clipped_vertices), std::move(clipped_faces));
  }

  const Mesh& clipped_mesh() const { return clipped_mesh_; }

 private:
  struct PointHash {
    std::size_t operator()(const Point& p) const noexcept {
      return boost::hash_range(p.begin(), p.end());
    }
  };

  static void clip(Triangle& tri, double threshold, std::vector<Triangle>& clipped) {
    auto interior = 0;
    auto boundary = 0;
    auto exterior = 0;

    for (auto v : tri.rowwise()) {
      if (v(0) < threshold) {
        interior++;
      } else if (v(0) == threshold) {
        boundary++;
      } else {
        exterior++;
      }
    }

    auto make_class = [](int i, int j, int k) constexpr -> int { return 16 * i + 4 * j + k; };
    auto append = [&clipped](const Triangle& t) {
      if (!degenerate(t)) {
        clipped.push_back(t);
      }
    };
    switch (make_class(interior, boundary, exterior)) {
      case make_class(1, 0, 2): {
        auto i = std::distance(tri.rowwise().begin(),
                               std::find_if(tri.rowwise().begin(), tri.rowwise().end(),
                                            [threshold](auto v) { return v(0) < threshold; }));
        tri = (tri({i, (i + 1) % 3, (i + 2) % 3}, Eigen::all)).eval();
        // Now vertices are ordered as (interior, exterior, exterior).
        auto t01 = (threshold - tri(0, 0)) / (tri(1, 0) - tri(0, 0));
        auto t02 = (threshold - tri(0, 0)) / (tri(2, 0) - tri(0, 0));
        Point p01 = tri.row(0) + t01 * (tri.row(1) - tri.row(0));
        Point p02 = tri.row(0) + t02 * (tri.row(2) - tri.row(0));
        p01(0) = threshold;
        p02(0) = threshold;
        append((Triangle() << tri.row(0), p01, p02).finished());
        break;
      }

      case make_class(1, 1, 1): {
        auto i = std::distance(tri.rowwise().begin(),
                               std::find_if(tri.rowwise().begin(), tri.rowwise().end(),
                                            [threshold](auto v) { return v(0) == threshold; }));
        tri = (tri({i, (i + 1) % 3, (i + 2) % 3}, Eigen::all)).eval();
        // Now vertices are ordered as either (boundary, interior, exterior)
        // or (boundary, exterior, interior).
        if (tri(1, 0) < tri(2, 0)) {
          // (boundary, interior, exterior).
          auto t12 = (threshold - tri(1, 0)) / (tri(2, 0) - tri(1, 0));
          Point p12 = tri.row(1) + t12 * (tri.row(2) - tri.row(1));
          p12(0) = threshold;
          append((Triangle() << tri.row(0), tri.row(1), p12).finished());
        } else {
          // (boundary, exterior, interior).
          auto t21 = (threshold - tri(2, 0)) / (tri(1, 0) - tri(2, 0));
          Point p21 = tri.row(2) + t21 * (tri.row(1) - tri.row(2));
          p21(0) = threshold;
          append((Triangle() << tri.row(0), p21, tri.row(2)).finished());
        }
        break;
      }

      case make_class(2, 0, 1): {
        auto i = std::distance(tri.rowwise().begin(),
                               std::find_if(tri.rowwise().begin(), tri.rowwise().end(),
                                            [threshold](auto v) { return v(0) > threshold; }));
        tri = (tri({i, (i + 1) % 3, (i + 2) % 3}, Eigen::all)).eval();
        // Now vertices are ordered as (exterior, interior, interior).
        auto t10 = (threshold - tri(1, 0)) / (tri(0, 0) - tri(1, 0));
        auto t20 = (threshold - tri(2, 0)) / (tri(0, 0) - tri(2, 0));
        Point p10 = tri.row(1) + t10 * (tri.row(0) - tri.row(1));
        Point p20 = tri.row(2) + t20 * (tri.row(0) - tri.row(2));
        p10(0) = threshold;
        p20(0) = threshold;
        // Delaunay triangulation.
        Vector normal = (tri.row(1) - tri.row(0)).cross(tri.row(2) - tri.row(0));
        auto [u, v] = plane_basis(normal);
        Point2 a{tri.row(1).dot(u), tri.row(1).dot(v)};
        Point2 b{tri.row(2).dot(u), tri.row(2).dot(v)};
        Point2 c{p20.dot(u), p20.dot(v)};
        Point2 d{p10.dot(u), p10.dot(v)};
        if (incircle_inexact(a, b, c, d) < 0.0) {
          append((Triangle() << tri.row(1), tri.row(2), p20).finished());
          append((Triangle() << tri.row(1), p20, p10).finished());
        } else {
          append((Triangle() << tri.row(1), tri.row(2), p10).finished());
          append((Triangle() << tri.row(2), p20, p10).finished());
        }
        break;
      }

      case make_class(1, 2, 0):
      case make_class(2, 1, 0):
      case make_class(3, 0, 0):
        append(tri);
        break;

      default:
        break;
    }
  }

  static bool degenerate(const Triangle& tri) {
    return tri.row(0) == tri.row(1) || tri.row(1) == tri.row(2) || tri.row(2) == tri.row(0);
  }

  static double incircle_inexact(const Point2& a, const Point2& b, const Point2& c,
                                 const Point2& d) {
    auto m00 = a(0) - d(0);
    auto m01 = a(1) - d(1);
    auto m02 = m00 * m00 + m01 * m01;
    auto m10 = b(0) - d(0);
    auto m11 = b(1) - d(1);
    auto m12 = m10 * m10 + m11 * m11;
    auto m20 = c(0) - d(0);
    auto m21 = c(1) - d(1);
    auto m22 = m20 * m20 + m21 * m21;
    Mat m;
    m << m00, m01, m02, m10, m11, m12, m20, m21, m22;
    return m.determinant();
  }

  static std::pair<Vector, Vector> plane_basis(const Vector& normal) {
    Vector u;
    if (normal(0) == 0.0) {
      u = Vector::UnitX();
    } else if (normal(1) == 0.0) {
      u = Vector::UnitY();
    } else if (normal(2) == 0.0) {
      u = Vector::UnitZ();
    } else {
      auto abs_nx = std::abs(normal(0));
      auto abs_ny = std::abs(normal(1));
      auto abs_nz = std::abs(normal(2));
      if (abs_nx <= abs_ny && abs_nx <= abs_nz) {
        u = {0.0, -normal(2), normal(1)};
      } else if (abs_ny <= abs_nx && abs_ny <= abs_nz) {
        u = {-normal(2), 0.0, normal(0)};
      } else {
        u = {-normal(1), normal(0), 0.0};
      }
    }

    Vector v = normal.cross(u);

    return {u.normalized(), v.normalized()};
  }

  Mesh clipped_mesh_;
};

inline Mesh clip(const Mesh& mesh, const geometry::Bbox3& bbox) {
  return MeshClipper(mesh, bbox).clipped_mesh();
}

}  // namespace polatory::isosurface
