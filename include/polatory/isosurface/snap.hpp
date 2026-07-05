#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>

namespace polatory::isosurface {

struct Stats {
  Index skipped{};    // beyond max_distance from the mesh
  Index satisfied{};  // already within tolerance of the snapped mesh, so not attempted
  Index dropped{};    // classified but could not be placed without self-intersection
  Index moved_vertices{};
  Index inserted_on_edges{};
  Index inserted_in_faces{};

  bool all_snapped_or_satisfied() const { return dropped == 0 && skipped == 0; }

  bool changed() const {
    return moved_vertices != 0 || inserted_on_edges != 0 || inserted_in_faces != 0;
  }
};

// Snaps the mesh to pass exactly through the given points, which become vertices. One pass; the
// pipeline re-applies it. See snapper/snapper.hpp.
Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               double resolution, const Mat3& aniso, double max_distance, Stats* stats = nullptr);

// Drops snapped vertices an earlier pass left redundant, by edge collapse, without moving any snap
// point beyond its tolerance of the surface. See snapper/thinner.hpp.
Mesh thin_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                       double resolution, const Mat3& aniso);

// Flattens the mesh by edge flips, never moving the surface beyond a point's tolerance.
// See snapper/smoother.hpp.
Mesh smooth_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                         double resolution, const Mat3& aniso);

}  // namespace polatory::isosurface
