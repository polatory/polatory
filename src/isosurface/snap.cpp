#include <polatory/isosurface/snap.hpp>

#include "snapper/smoother.hpp"
#include "snapper/snapper.hpp"
#include "snapper/thinner.hpp"

namespace polatory::isosurface {

Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               double resolution, const geometry::Bbox3& bbox, double max_distance,
               const Mat3& aniso, Stats* stats) {
  if (stats != nullptr) {
    *stats = {};
  }
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  snapper::Snapper snapper(mesh, points, tolerances, resolution, bbox, max_distance, aniso);
  if (stats != nullptr) {
    *stats = snapper.stats();
  }
  return std::move(snapper).result();
}

Mesh thin_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                       double resolution, const Mat3& aniso) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  return snapper::Thinner(mesh, points, tolerances, resolution, aniso).result();
}

Mesh smooth_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                         double resolution, const Mat3& aniso) {
  constexpr double kDegree = 0.017453292519943295;
  constexpr double kMinAngle = 5.0 * kDegree;
  return snapper::Smoother(mesh, points, tolerances, resolution, aniso, kMinAngle).result();
}

}  // namespace polatory::isosurface
