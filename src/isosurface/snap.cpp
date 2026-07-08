#include <polatory/isosurface/snap.hpp>

#include "snapper/edge_optimizer.hpp"
#include "snapper/smoother.hpp"
#include "snapper/snapper.hpp"

namespace polatory::isosurface {

Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               double resolution, const Mat3& aniso, Stats* stats, bool move_nearby_vertices) {
  if (stats != nullptr) {
    *stats = {};
  }
  if (points.rows() == 0 || mesh.faces().rows() == 0) {
    return mesh;
  }

  snapper::Snapper snapper(mesh, points, tolerances, resolution, aniso, move_nearby_vertices);
  if (stats != nullptr) {
    *stats = snapper.stats();
  }
  return std::move(snapper).result();
}

Mesh optimize_edges(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                    double resolution, const Mat3& aniso) {
  if (mesh.faces().rows() == 0) {
    return mesh;
  }

  return snapper::EdgeOptimizer(mesh, points, tolerances, resolution, aniso).result();
}

Mesh smooth_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                         double resolution, const Mat3& aniso) {
  return snapper::Smoother(mesh, points, tolerances, resolution, aniso).result();
}

}  // namespace polatory::isosurface
