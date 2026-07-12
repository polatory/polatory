#pragma once

#include <Eigen/Core>
#include <array>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/types.hpp>
#include <vector>

#include "utility.hpp"

namespace polatory::isosurface {

// Places a vertex that merges or seals the surface around `triangles` (the incident faces): the
// area-weighted quadric minimizer  x = argmin_x  sum_t area_t (n_t.x + d_t)^2  -- keeping a
// crease/corner instead of averaging it away. The fit is in the aniso-transformed frame so it
// respects the anisotropic resolution. Directions the planes leave free (a flat patch) keep
// `anchor`'s centroid; x is finally clamped into lattice node `node`'s cell so it stays nearest to
// that node. `anchor` and `triangles` are untransformed positions.
inline geometry::Point3 quadric_position(
    const geometry::Points3& anchor, const std::vector<std::array<geometry::Point3, 3>>& triangles,
    const Mat3& aniso, const Mat3& aniso_inv, const rmt::PrimitiveLattice& lattice,
    const rmt::LatticeCoordinates& node) {
  using geometry::Point3;
  using geometry::Points3;
  using geometry::Vector3;

  Points3 a_anchor = geometry::transform_points<3>(aniso, anchor);
  Point3 centroid = a_anchor.colwise().mean();

  // Accumulate a and b, the matrix and vector of that energy's normal equation a x = b.
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

  Point3 x = quadric_minimize(a, b, centroid, 1e-3);
  return lattice.clamp_to_node(geometry::transform_point<3>(aniso_inv, x), node);
}

}  // namespace polatory::isosurface
