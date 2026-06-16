#include <polatory/isosurface/snap.hpp>
#include <polatory/isosurface/snapper/snapper.hpp>

namespace polatory::isosurface {

Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const geometry::Bbox3& bbox,
               double min_distance, double max_distance, const Mat3& aniso) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  snapper::Snapper s(mesh.vertices(), mesh.faces(), bbox, min_distance, max_distance, aniso);
  return s.snap(points);
}

}  // namespace polatory::isosurface
