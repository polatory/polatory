#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Projects mesh vertices onto the field's level set f = isovalue, committing each move only if it
// introduces no self-intersection.
Mesh refine_vertices(const Mesh& mesh, const FieldFunction& field_fn, double isovalue,
                     const geometry::Bbox3& bbox, double resolution, const Mat3& aniso);

}  // namespace polatory::isosurface
