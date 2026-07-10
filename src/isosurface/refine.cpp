#include <polatory/isosurface/refine.hpp>

#include "vertex_refiner.hpp"

namespace polatory::isosurface {

Mesh refine_vertices(const Mesh& mesh, const FieldFunction& field_fn, double isovalue,
                     const geometry::Bbox3& bbox, double resolution, const Mat3& aniso) {
  if (mesh.is_empty()) {
    return mesh;
  }
  return VertexRefiner(mesh, field_fn, isovalue, bbox, resolution, aniso).result();
}

}  // namespace polatory::isosurface
