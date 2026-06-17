#include <cstdlib>
#include <polatory/isosurface/smooth.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/isosurface/snapper/snapper.hpp>

namespace polatory::isosurface {

Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               const geometry::Bbox3& bbox, double max_distance, const Mat3& aniso) {
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  snapper::Snapper s(mesh.vertices(), mesh.faces(), bbox, max_distance, aniso);
  auto out = s.snap(points, tolerances);

  // EXPERIMENT (POLATORY_SMOOTH): flatten cusp/sliver artifacts by edge flips that lower the
  // worst dihedral, without moving any vertex (so all snapped points stay honored).
  if (std::getenv("POLATORY_SMOOTH") != nullptr) {
    out = smooth_by_flips(out, aniso);
  }

  return out;
}

}  // namespace polatory::isosurface
