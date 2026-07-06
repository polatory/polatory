#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>

namespace polatory::isosurface {

// The three counts partition the snap points.
struct Stats {
  Index skipped{};     // no candidate: the nearest face was beyond the resolution
  Index honored{};     // ended within tolerance of the surface
  Index dishonored{};  // within the resolution but left off the surface (contention, or unplaceable
                       // without self-intersection)
};

// Snaps the mesh to pass exactly through the given points, which become vertices. One pass; the
// pipeline re-applies it. See snapper/snapper.hpp.
Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               double resolution, const Mat3& aniso, Stats* stats = nullptr);

// Drops snapped vertices an earlier pass left redundant, by edge collapse, without moving any snap
// point beyond its tolerance of the surface. See snapper/thinner.hpp.
Mesh thin_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                       double resolution, const Mat3& aniso);

// Flattens the mesh by edge flips, never moving the surface beyond a point's tolerance.
// See snapper/smoother.hpp.
Mesh smooth_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                         double resolution, const Mat3& aniso);

}  // namespace polatory::isosurface
