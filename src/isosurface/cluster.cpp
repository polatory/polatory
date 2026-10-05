#include <jizai/isosurface/cluster.hpp>

#include "vertex_clusterer.hpp"

namespace jizai::isosurface {

Mesh cluster_vertices(const Mesh& mesh, const rmt::PrimitiveLattice& lattice, const Mat3& aniso) {
  return VertexClusterer(mesh, lattice, aniso).result();
}

}  // namespace jizai::isosurface
