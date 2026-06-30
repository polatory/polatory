#include <polatory/isosurface/cluster.hpp>

#include "vertex_clusterer.hpp"

namespace polatory::isosurface {

Mesh cluster_vertices(const Mesh& mesh, const rmt::PrimitiveLattice& lattice, const Mat3& aniso) {
  return VertexClusterer(mesh, lattice, aniso).result();
}

}  // namespace polatory::isosurface
