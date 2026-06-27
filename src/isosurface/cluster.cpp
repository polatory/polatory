#include <polatory/isosurface/cluster.hpp>

#include "genus_reducer.hpp"
#include "vertex_clusterer.hpp"

namespace polatory::isosurface {

Mesh cluster_mesh_vertices(const Mesh& mesh, const rmt::PrimitiveLattice& lattice,
                           const Mat3& aniso) {
  return VertexClusterer(mesh, lattice, aniso).result();
}

Mesh reduce_genus(const Mesh& mesh, const rmt::PrimitiveLattice& lattice, const Mat3& aniso) {
  return GenusReducer(mesh, lattice, aniso).result();
}

}  // namespace polatory::isosurface
