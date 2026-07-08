#include <polatory/isosurface/relax.hpp>

#include "vertex_relaxer.hpp"

namespace polatory::isosurface {

Mesh relax_vertices(const Mesh& mesh, const FieldFunction& field_fn, double isovalue,
                    double resolution, const Mat3& aniso, int passes, const geometry::Bbox3& bbox) {
  if (mesh.is_empty()) {
    return mesh;
  }
  return VertexRelaxer(mesh, field_fn, isovalue, resolution, aniso, passes, bbox).result();
}

}  // namespace polatory::isosurface
