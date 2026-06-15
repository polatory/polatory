#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/clip.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice.hpp>
#include <polatory/isosurface/sign.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <unordered_set>

namespace polatory::isosurface {

class Isosurface {
 public:
  Isosurface(const geometry::Bbox3& bbox, double resolution)
      : Isosurface(bbox, resolution, Mat3::Identity()) {}

  Isosurface(const geometry::Bbox3& bbox, double resolution, const Mat3& aniso)
      : lattice_(bbox, resolution, aniso), aniso_(aniso) {
    if (!(aniso.determinant() > 0.0)) {
      throw std::invalid_argument("aniso must have a positive determinant");
    }
  }

  void clear() { lattice_.clear(); }

  Mesh generate(FieldFunction& field_fn, double isovalue = 0.0, int refine = 1) {
    if (refine < 0) {
      throw std::runtime_error("refine must be non-negative");
    }

    field_fn.set_evaluation_bbox(lattice_.second_extended_bbox());

    lattice_.add_all_nodes(field_fn, isovalue);
    lattice_.refine_vertices(field_fn, isovalue, refine);

    return generate_common();
  }

  Mesh generate_from_seed_points(const geometry::Points3& seed_points, FieldFunction& field_fn,
                                 double isovalue = 0.0, int refine = 1) {
    if (seed_points.rows() == 0) {
      throw std::runtime_error("seed points must not be empty");
    }

    if (refine < 0) {
      throw std::runtime_error("refine must be non-negative");
    }

    field_fn.set_evaluation_bbox(lattice_.second_extended_bbox());

    lattice_.add_nodes_from_seed_points(seed_points, field_fn, isovalue);
    lattice_.refine_vertices(field_fn, isovalue, refine);

    return generate_common();
  }

  // Sets the points that the generated mesh will be snapped to so that it passes
  // exactly through them. Points whose projection falls outside the extended bbox
  // are ignored, which keeps the mesh boundary intact.
  //
  // max_distance_ratio bounds how far a point may lie from the mesh to be snapped,
  // as a fraction of the mesh resolution. It must be in (0, 1].
  void set_snap_points(const geometry::Points3& points, double max_distance_ratio = 0.5) {
    if (!(max_distance_ratio > 0.0 && max_distance_ratio <= 1.0)) {
      throw std::invalid_argument("snap max distance ratio must be in (0, 1]");
    }

    snap_points_ = points;
    snap_max_distance_ratio_ = max_distance_ratio;
  }

 private:
  Mesh generate_common() {
    lattice_.cluster_vertices();

    Mesh mesh;

    // Unclustering non-manifold vertices may require a few iterations.
    while (true) {
      mesh = lattice_.get_mesh();
      MeshDefectsFinder defects(mesh);

      auto vis = defects.singular_vertices();
      auto fis = defects.intersecting_faces();

      std::unordered_set<Index> vertices_to_uncluster;
      for (auto vi : vis) {
        vertices_to_uncluster.insert(vi);
      }
      for (auto fi : fis) {
        auto f = mesh.faces().row(fi);
        vertices_to_uncluster.insert(f(0));
        vertices_to_uncluster.insert(f(1));
        vertices_to_uncluster.insert(f(2));
      }

      auto num_unclustered = lattice_.uncluster_vertices(vertices_to_uncluster);
      if (num_unclustered == 0) {
        break;
      }
    }

    // Snap the mesh to the points before clipping. Restricting snapping to points
    // whose projection stays inside first_extended_bbox keeps it away from the mesh
    // boundary, which lies further out (near second_extended_bbox); the clip then
    // produces the clean on-bbox boundary.
    if (snap_points_.rows() > 0 && !mesh.is_empty()) {
      auto max_distance = snap_max_distance_ratio_ * lattice_.resolution();
      mesh = snap_mesh(mesh, snap_points_, lattice_.first_extended_bbox(), max_distance, aniso_);
    }

    mesh = clip(mesh, lattice_.bbox());

    if (mesh.is_empty() &&
        lattice_.value_sign_at_arbitrary_point_within_bbox() == BinarySign::kNeg) {
      mesh = Mesh(EntireTag{});
    }

    lattice_.clear();

    return mesh;
  }

  rmt::Lattice lattice_;
  Mat3 aniso_;
  geometry::Points3 snap_points_;
  double snap_max_distance_ratio_{0.5};
};

}  // namespace polatory::isosurface
