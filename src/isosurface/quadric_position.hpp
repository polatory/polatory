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

  // Accumulate aa and bb, the matrix and vector of that energy's normal equation.
  Mat3 aa = Mat3::Zero();
  Vector3 bb = Vector3::Zero();
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
    auto d = -n.dot(p0);
    aa += w * n.transpose() * n;
    bb += w * d * n;
  }

  // Solve aa x = -bb in a rank-revealing manner: move off the centroid only when aa constrains more
  // than one direction -- a crease (rank 2) or corner (rank 3) -- and along just those. A flat
  // patch (rank 1) stays at the centroid.
  Eigen::SelfAdjointEigenSolver<Mat3> es(aa);
  auto floor = 1e-3 * es.eigenvalues()(2);
  Vector3 y = Vector3::Zero();
  if (es.eigenvalues()(1) > floor) {
    Vector3 r = -(centroid * aa + bb);
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
