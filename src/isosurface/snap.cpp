#include <polatory/isosurface/snap.hpp>
#include "snapper/smoother.hpp"
#include "snapper/snapper.hpp"
#include "snapper/thinner.hpp"
#include <utility>

namespace polatory::isosurface {

Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               const geometry::Bbox3& bbox, double max_distance, const Mat3& aniso) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  return snapper::Snapper(mesh, points, tolerances, bbox, max_distance, aniso).result();
}

Mesh thin_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                       double resolution, const Mat3& aniso) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  return snapper::Thinner(mesh, points, tolerances, resolution, aniso).result();
}

Mesh smooth_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                         double resolution, const Mat3& aniso, std::vector<bool> snapped) {
  constexpr double kDegree = 0.017453292519943295;
  constexpr double kMinAngle = 5.0 * kDegree;
  return snapper::Smoother(mesh, points, tolerances, resolution, aniso, kMinAngle,
                           std::move(snapped))
      .result();
}

}  // namespace polatory::isosurface
