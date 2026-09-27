#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <array>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

inline geometry::Point3 quadric_position(
    const geometry::Points3& vertices,
    const std::vector<std::array<geometry::Point3, 3>>& triangles, const Mat3& aniso,
    const Mat3& aniso_inv, const rmt::PrimitiveLattice& lattice,
    const rmt::LatticeCoordinates& node) {
  using geometry::Point3;
  using geometry::Points3;
  using geometry::Vector3;

  Points3 a_vertices = geometry::transform_points<3>(aniso, vertices);
  Point3 centroid = a_vertices.colwise().mean();

  Mat3 a = Mat3::Zero();
  Vector3 b = Vector3::Zero();
  for (const auto& t : triangles) {
    Point3 p0 = geometry::transform_point<3>(aniso, t.at(0));
    Point3 p1 = geometry::transform_point<3>(aniso, t.at(1));
    Point3 p2 = geometry::transform_point<3>(aniso, t.at(2));
    Vector3 n = (p1 - p0).cross(p2 - p0);
    auto w = n.norm();
    if (w == 0.0) {
      continue;
    }
    n /= w;
    a += w * n.transpose() * n;
    b += w * n.dot(p0) * n;
  }

  Eigen::SelfAdjointEigenSolver<Mat3> es(a);
  auto floor = 1e-3 * es.eigenvalues()(2);
  Vector3 y = Vector3::Zero();
  // Stay at the centroid unless the planes form a crease (rank 2) or a corner (rank 3).
  if (es.eigenvalues()(1) > floor) {
    Vector3 r = b - centroid * a;  // a is symmetric
    for (auto k = 0; k < 3; k++) {
      auto eval = es.eigenvalues()(k);
      if (eval > floor) {
        Vector3 evec = es.eigenvectors().col(k).transpose();
        y += (r.dot(evec) / eval) * evec;
      }
    }
  }
  Point3 x = centroid + y;
  return lattice.clamp_to_node(geometry::transform_point<3>(aniso_inv, x), node);
}

}  // namespace polatory::isosurface
