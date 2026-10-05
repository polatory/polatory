#pragma once

#include <jizai/geometry/bbox3d.hpp>
#include <jizai/isosurface/field_function.hpp>
#include <jizai/isosurface/mesh.hpp>
#include <jizai/types.hpp>

namespace jizai::isosurface {

// Moves each vertex toward the isosurface, unless the move introduces self-intersection.
Mesh refine_vertices(const Mesh& mesh, const FieldFunction& field_fn, double isovalue,
                     const geometry::Bbox3& bbox, double resolution, const Mat3& aniso);

}  // namespace jizai::isosurface
