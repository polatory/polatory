#pragma once

#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Projects mesh vertices onto the field's level set f = isovalue, committing each move only if it
// introduces no self-intersection. See vertex_refiner.hpp.
Mesh refine_vertices(const Mesh& mesh, const FieldFunction& field_fn, double isovalue,
                     double resolution, const Mat3& aniso);

}  // namespace polatory::isosurface
