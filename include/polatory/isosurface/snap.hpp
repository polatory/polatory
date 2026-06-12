#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>

namespace polatory::isosurface {

// Snaps the mesh so that it passes exactly through the given points, which then
// become vertices of the mesh.
//
// A point is snapped only if its distance to the mesh is at most max_distance and
// both the point and its projection onto the mesh lie inside bbox. Provided bbox
// contains no mesh boundary in its interior, the mesh boundary is left untouched.
// See snapper.hpp for details. Passing no points returns the mesh unchanged.
Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const geometry::Bbox3& bbox,
               double max_distance);

}  // namespace polatory::isosurface
