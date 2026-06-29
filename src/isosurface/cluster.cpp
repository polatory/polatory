#include <polatory/isosurface/cluster.hpp>

#include "defeaturer.hpp"
#include "vertex_clusterer.hpp"

namespace polatory::isosurface {

Mesh cluster_mesh_vertices(const Mesh& mesh, const rmt::PrimitiveLattice& lattice,
                           const Mat3& aniso) {
  return VertexClusterer(mesh, lattice, aniso).result();
}

Mesh defeature(const Mesh& mesh, const rmt::PrimitiveLattice& lattice) {
  return Defeaturer(mesh, lattice).result();
}

}  // namespace polatory::isosurface
