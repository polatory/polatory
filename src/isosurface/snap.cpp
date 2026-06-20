#include <polatory/isosurface/snap.hpp>
#include <polatory/isosurface/snapper/smoother.hpp>
#include <polatory/isosurface/snapper/snapper.hpp>
#include <polatory/isosurface/snapper/thinner.hpp>
#include <utility>

namespace polatory::isosurface {

Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               const geometry::Bbox3& bbox, double max_distance, const Mat3& aniso) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  snapper::Snapper s(mesh.vertices(), mesh.faces(), bbox, max_distance, aniso);
  return s.snap(points, tolerances);
}

Mesh thin_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                       const Mat3& aniso) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  snapper::Thinner t(mesh.vertices(), mesh.faces(), points, tolerances, aniso);
  return t.thin();
}

Mesh smooth_snapped_mesh(const Mesh& mesh, double resolution, const Mat3& aniso,
                         std::vector<bool> snapped) {
  constexpr double kDegree = 0.017453292519943295;  // radians
  constexpr double kMinAngle = 10.0 * kDegree;

  return snapper::Smoother(mesh, resolution, aniso, kMinAngle, std::move(snapped)).mesh();
}

}  // namespace polatory::isosurface
