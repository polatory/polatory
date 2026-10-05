#pragma once

#include <jizai/isosurface/mesh.hpp>
#include <jizai/isosurface/rmt/primitive_lattice.hpp>
#include <jizai/types.hpp>

namespace jizai::isosurface {

// Merges the vertices of each lattice node into one, unless the merge makes the mesh non-manifold
// or self-intersecting.
Mesh cluster_vertices(const Mesh& mesh, const rmt::PrimitiveLattice& lattice, const Mat3& aniso);

}  // namespace jizai::isosurface
