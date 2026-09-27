#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>

namespace polatory::isosurface {

struct Stats {
  Index skipped{};     // farther than the resolution from the mesh
  Index honored{};     // within the tolerance of the mesh
  Index dishonored{};  // otherwise
};

// Moves or inserts vertices so that the mesh passes through the points.
Mesh snap_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
               double resolution, const Mat3& aniso, Stats* stats = nullptr);

// Removes redundant snapped vertices by edge collapses.
Mesh thin_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                       double resolution, const Mat3& aniso);

// Reduces the total dihedral angle by edge flips.
Mesh smooth_snapped_mesh(const Mesh& mesh, const geometry::Points3& points, const VecX& tolerances,
                         double resolution, const Mat3& aniso);

}  // namespace polatory::isosurface
