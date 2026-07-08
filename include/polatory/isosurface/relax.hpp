#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/field_function.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Regularizes triangle shape by `passes` sweeps of tangential vertex relaxation onto the field's
// level set f = isovalue, pinning sharp features and the mesh boundary. A vertex a relaxation slides
// very close to the bbox is snapped exactly onto it (guarded), so a later clip cuts it cleanly. See
// vertex_relaxer.hpp.
Mesh relax_vertices(const Mesh& mesh, const FieldFunction& field_fn, double isovalue,
                    double resolution, const Mat3& aniso, int passes, const geometry::Bbox3& bbox);

}  // namespace polatory::isosurface
