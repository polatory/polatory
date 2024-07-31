#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <iterator>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/surface.hpp>
#include <polatory/isosurface/types.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface {

class surface_clipper {
  using Triangle = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
  using Faces = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

 public:
  surface_clipper(const surface& s, const geometry::bbox3d& bbox) {
    const auto& vertices = s.vertices();
    const auto& faces = s.faces();

    std::vector<Triangle> triangles;
    for (auto f : faces.rowwise()) {
      geometry::point3d p = vertices.row(f(0));
      geometry::point3d q = vertices.row(f(1));
      geometry::point3d r = vertices.row(f(2));
      triangles.push_back((Triangle() << p, q, r).finished());
    }

    std::array<geometry::matrix3d, 6> permutations{
        (geometry::matrix3d() << 1, 0, 0, 0, 1, 0, 0, 0, 1).finished(),   // x, y, z
        (geometry::matrix3d() << -1, 0, 0, 0, 0, 1, 0, 1, 0).finished(),  // -x, z, y
        (geometry::matrix3d() << 0, 1, 0, 0, 0, 1, 1, 0, 0).finished(),   // y, z, x
        (geometry::matrix3d() << 0, -1, 0, 1, 0, 0, 0, 0, 1).finished(),  // -y, x, z
        (geometry::matrix3d() << 0, 0, 1, 1, 0, 0, 0, 1, 0).finished(),   // z, x, y
        (geometry::matrix3d() << 0, 0, -1, 0, 1, 0, 1, 0, 0).finished()   // -z, y, x
    };
    std::array<double, 6> thresholds{bbox.max()(0),  -bbox.min()(0), bbox.max()(1),
                                     -bbox.min()(1), bbox.max()(2),  -bbox.min()(2)};

    std::vector<Triangle> clipped;
    for (auto face = 0; face < 6; face++) {
      const auto& perm = permutations.at(face);
      auto threshold = thresholds.at(face);
      for (auto& tri : triangles) {
        tri = geometry::transform_points<3>(perm, tri);
        clip_triangle(tri, threshold, clipped);
      }
      for (auto& tri : clipped) {
        tri = geometry::transform_points<3>(perm.transpose(), tri);
      }
      std::swap(triangles, clipped);
      clipped.clear();
    }

    std::unordered_map<geometry::point3d, index_t, point_hash> vertex_map;
    for (const auto& tri : triangles) {
      for (auto v : tri.rowwise()) {
        if (!vertex_map.contains(v)) {
          vertex_map.emplace(v, static_cast<index_t>(vertex_map.size()));
        }
      }
    }

    geometry::points3d clipped_vertices(static_cast<index_t>(vertex_map.size()), 3);
    Faces clipped_faces(static_cast<index_t>(triangles.size()), 3);

    for (const auto& pair : vertex_map) {
      clipped_vertices.row(pair.second) = pair.first;
    }

    for (std::size_t i = 0; i < triangles.size(); i++) {
      const auto& tri = triangles.at(i);
      auto face = clipped_faces.row(static_cast<index_t>(i));
      face(0) = vertex_map.at(tri.row(0));
      face(1) = vertex_map.at(tri.row(1));
      face(2) = vertex_map.at(tri.row(2));
    }

    clipped_surface_ = surface(clipped_vertices, clipped_faces);
  }

  const surface& clipped_surface() const { return clipped_surface_; }

 private:
  struct point_hash {
    std::size_t operator()(const geometry::point3d& p) const noexcept {
      return boost::hash_range(p.begin(), p.end());
    }
  };

  static constexpr int make_class(int interior, int boundary, int exterior) {
    return 100 * interior + 10 * boundary + exterior;
  }

  static std::pair<geometry::vector3d, geometry::vector3d> plane_basis(
      const geometry::vector3d& normal) {
    geometry::vector3d u;
    if (normal(0) == 0.0) {
      u = geometry::vector3d::UnitX();
    } else if (normal(1) == 0.0) {
      u = geometry::vector3d::UnitY();
    } else if (normal(2) == 0.0) {
      u = geometry::vector3d::UnitZ();
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

    geometry::vector3d v = normal.cross(u);

    return {u.normalized(), v.normalized()};
  }

  static double incircle2d_inexact(const geometry::point2d& a, const geometry::point2d& b,
                                   const geometry::point2d& c, const geometry::point2d& d) {
    auto m00 = a(0) - d(0);
    auto m01 = a(1) - d(1);
    auto m02 = m00 * m00 + m01 * m01;
    auto m10 = b(0) - d(0);
    auto m11 = b(1) - d(1);
    auto m12 = m10 * m10 + m11 * m11;
    auto m20 = c(0) - d(0);
    auto m21 = c(1) - d(1);
    auto m22 = m20 * m20 + m21 * m21;
    return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) +
           m02 * (m10 * m21 - m11 * m20);
  }

  static void clip_triangle(Triangle& tri, double threshold, std::vector<Triangle>& clipped) {
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

    switch (make_class(interior, boundary, exterior)) {
      case make_class(1, 0, 2): {
        auto i = std::distance(tri.rowwise().begin(),
                               std::find_if(tri.rowwise().begin(), tri.rowwise().end(),
                                            [threshold](auto v) { return v(0) < threshold; }));
        tri = (tri({i, (i + 1) % 3, (i + 2) % 3}, Eigen::all)).eval();
        // Now vertices are ordered as (interior, exterior, exterior).
        auto t01 = (threshold - tri(0, 0)) / (tri(1, 0) - tri(0, 0));
        auto t02 = (threshold - tri(0, 0)) / (tri(2, 0) - tri(0, 0));
        geometry::point3d p01 = tri.row(0) + t01 * (tri.row(1) - tri.row(0));
        geometry::point3d p02 = tri.row(0) + t02 * (tri.row(2) - tri.row(0));
        clipped.push_back((Triangle() << tri.row(0), p01, p02).finished());
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
          geometry::point3d p12 = tri.row(1) + t12 * (tri.row(2) - tri.row(1));
          clipped.push_back((Triangle() << tri.row(0), tri.row(1), p12).finished());
        } else {
          // (boundary, exterior, interior).
          auto t21 = (threshold - tri(2, 0)) / (tri(1, 0) - tri(2, 0));
          geometry::point3d p21 = tri.row(2) + t21 * (tri.row(1) - tri.row(2));
          clipped.push_back((Triangle() << tri.row(0), p21, tri.row(2)).finished());
        }
        break;
      }

      case make_class(1, 2, 0):
        clipped.push_back(tri);
        break;

      case make_class(2, 0, 1): {
        auto i = std::distance(tri.rowwise().begin(),
                               std::find_if(tri.rowwise().begin(), tri.rowwise().end(),
                                            [threshold](auto v) { return v(0) > threshold; }));
        tri = (tri({i, (i + 1) % 3, (i + 2) % 3}, Eigen::all)).eval();
        // Now vertices are ordered as (exterior, interior, interior).
        auto t10 = (threshold - tri(1, 0)) / (tri(0, 0) - tri(1, 0));
        auto t20 = (threshold - tri(2, 0)) / (tri(0, 0) - tri(2, 0));
        geometry::point3d p10 = tri.row(1) + t10 * (tri.row(0) - tri.row(1));
        geometry::point3d p20 = tri.row(2) + t20 * (tri.row(0) - tri.row(2));
        // Delaunay triangulation.
        geometry::vector3d normal = (tri.row(1) - tri.row(0)).cross(tri.row(2) - tri.row(0));
        auto [u, v] = plane_basis(normal);
        geometry::point2d a{tri.row(1).dot(u), tri.row(1).dot(v)};
        geometry::point2d b{tri.row(2).dot(u), tri.row(2).dot(v)};
        geometry::point2d c{p20.dot(u), p20.dot(v)};
        geometry::point2d d{p10.dot(u), p10.dot(v)};
        if (incircle2d_inexact(a, b, c, d) < 0.0) {
          clipped.push_back((Triangle() << tri.row(1), tri.row(2), p20).finished());
          clipped.push_back((Triangle() << tri.row(1), p20, p10).finished());
        } else {
          clipped.push_back((Triangle() << tri.row(1), tri.row(2), p10).finished());
          clipped.push_back((Triangle() << tri.row(2), p20, p10).finished());
        }
        break;
      }

      case make_class(2, 1, 0):
        clipped.push_back(tri);
        break;

      case make_class(3, 0, 0):
        clipped.push_back(tri);
        break;

      default:
        break;
    }
  }

  surface clipped_surface_;
};

inline surface clip_surface(const surface& s, const geometry::bbox3d& bbox) {
  return surface_clipper(s, bbox).clipped_surface();
}

}  // namespace polatory::isosurface
