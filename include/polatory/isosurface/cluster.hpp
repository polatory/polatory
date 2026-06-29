#pragma once

#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/types.hpp>

namespace polatory::isosurface {

// Clusters each lattice node's vertices into a single vertex where doing so stays manifold.
// See vertex_clusterer.hpp.
Mesh cluster_mesh_vertices(const Mesh& mesh, const rmt::PrimitiveLattice& lattice,
                           const Mat3& aniso);

// Removes sub-resolution topological artifacts (annulus regions) from the RMT surface. See
// defeaturer.hpp.
Mesh defeature(const Mesh& mesh, const rmt::PrimitiveLattice& lattice);

}  // namespace polatory::isosurface
