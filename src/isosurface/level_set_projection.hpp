#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

namespace polatory::isosurface {

// One Newton step per vertex toward the level set f = isovalue, along the field gradient estimated
// numerically from a regular-tetrahedron stencil. The step is capped in the aniso frame so an
// ill-conditioned fit cannot bolt a vertex across the mesh. Returns only the vertices that move,
// each paired with its target position. SDFs are only C0 (a crease is a gradient discontinuity), so
// the gradient is always taken by finite differences, never analytically.
inline std::vector<std::pair<Index, geometry::Point3>> project_to_level_set(
    const FieldFunction& field_fn, double isovalue, const geometry::Points3& positions,
    double resolution, const Mat3& aniso) {
  using geometry::Point3;
  using geometry::Points3;
  using geometry::Vector3;

  constexpr double kMaxMoveRatio = 0.5;   // per-step cap, in units of resolution
  constexpr double kMinMoveRatio = 1e-5;  // below this a vertex is already on the level set
  auto nv = positions.rows();

  auto d = 1e-3 * resolution;
  // Regular-tetrahedron offsets (directions prescaled by d): sum to zero and sum of outer products
  // is 4 d^2 I, so the linear fit is the closed form below.
  Mat<4, 3> s;
  s << Vector3{d, d, d}, Vector3{d, -d, -d}, Vector3{-d, d, -d}, Vector3{-d, -d, d};
  Points3 samples(4 * nv, 3);
  for (Index i = 0; i < nv; i++) {
    samples.middleRows(4 * i, 4) = s.rowwise() + positions.row(i);
  }
  VecX v = field_fn(samples).array() - isovalue;

  std::vector<std::pair<Index, Point3>> moves;
  auto max_move = kMaxMoveRatio * resolution;
  auto min_move = kMinMoveRatio * resolution;
  // The least-squares gradient of the linear fit: g = (s^T s)^-1 s^T v = s^T v / (4 d^2).
  auto inv_4dd = 1.0 / (4.0 * d * d);
  for (Index i = 0; i < nv; i++) {
    auto vs = v.segment(4 * i, 4);
    Vector3 grad = inv_4dd * vs.transpose() * s;
    auto gn2 = grad.squaredNorm();
    if (gn2 == 0.0) {
      continue;
    }
    auto value = 0.25 * vs.sum();
    Vector3 step = (value / gn2) * grad;
    auto a_len = geometry::transform_vector<3>(aniso, step).norm();
    if (!(a_len >= min_move)) {
      // Skips a vertex already on the level set, and also a non-finite step (NaN fails every
      // comparison, so it must be rejected here rather than passed to the caps below).
      continue;
    }
    if (a_len > max_move) {
      step *= max_move / a_len;
    }
    moves.emplace_back(i, Point3(positions.row(i) - step));
  }
  return moves;
}

}  // namespace polatory::isosurface
