#pragma once

#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Merges the vertices of each lattice node into one, unless the merge makes the mesh non-manifold
// or self-intersecting.
Mesh cluster_vertices(const Mesh& mesh, const rmt::PrimitiveLattice& lattice, const Mat3& aniso);

}  // namespace polatory::isosurface
