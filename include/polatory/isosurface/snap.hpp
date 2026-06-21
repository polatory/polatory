#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <vector>

namespace polatory::isosurface {

// Snaps the mesh to pass exactly through the given points, which become vertices. One pass; the
// pipeline re-applies it. See snapper/snapper.hpp.
Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               const geometry::Bbox3& bbox, double max_distance, const Mat3& aniso);

// Drops snapped vertices an earlier pass left redundant, by edge collapse, without moving any snap
// point beyond its tolerance of the surface. See snapper/thinner.hpp.
Mesh thin_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                       double resolution, const Mat3& aniso);

// Flattens the snapped region by edge flips, never moving the surface beyond a point's tolerance.
// See snapper/smoother.hpp.
Mesh smooth_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                         double resolution, const Mat3& aniso, std::vector<bool> snapped);

}  // namespace polatory::isosurface
