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

  // Sets the points the generated mesh is snapped to so it passes exactly through them. All
  // distances are fractions of the mesh resolution. Points projecting outside the extended bbox are
  // ignored, keeping the boundary intact. relative_distance bounds how far a point may lie from the
  // mesh to snap. relative_tolerances (if non-empty) is the slack each point's surface may keep: a
  // point already within tolerance is skipped and a redundant inserted vertex within tolerance is
  // thinned, so a dense polyline does not over-triangulate. max_passes re-applies snapping to
  // recover points that lost contention, and only matters with a positive tolerance.
  void set_snap_points(const geometry::Points3& points, double relative_distance = 0.5,
                       const VecX& relative_tolerances = VecX(), int max_passes = 8) {
    if (!(relative_distance > 0.0 && relative_distance <= 1.0)) {
      throw std::invalid_argument("snap relative distance must be in (0, 1]");
    }
    if (relative_tolerances.size() != 0) {
      if (relative_tolerances.size() != points.rows()) {
        throw std::invalid_argument("snap relative tolerances must have one entry per point");
      }
      if (!(relative_tolerances.minCoeff() >= 0.0 &&
            relative_tolerances.maxCoeff() <= relative_distance)) {
        throw std::invalid_argument("snap relative tolerances must be in [0, relative distance]");
      }
    }
    if (max_passes < 1) {
      throw std::invalid_argument("snap max passes must be at least 1");
    }

    snap_points_ = points;
    rel_snap_dist_ = relative_distance;
    rel_snap_tols_ = relative_tolerances;
    snap_max_passes_ = max_passes;
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

    auto same = [](const Mesh& a, const Mesh& b) {
      const auto& av = a.vertices();
      const auto& bv = b.vertices();
      const auto& af = a.faces();
      const auto& bf = b.faces();
      return av.rows() == bv.rows() && af.rows() == bf.rows() && (av.array() == bv.array()).all() &&
             (af.array() == bf.array()).all();
    };

    // Snap before clipping, re-applying until a pass is a no-op: a point that lost contention can
    // snap to the finer mesh a later pass leaves. first_extended_bbox keeps snapping off the
    // boundary.
    auto res = lattice_.resolution();
    if (snap_points_.rows() > 0 && !mesh.is_empty()) {
      VecX tols = res * rel_snap_tols_;
      int passes = tols.size() != 0 && tols.maxCoeff() > 0.0 ? snap_max_passes_ : 1;
      for (int pass = 0; pass < passes; pass++) {
        Mesh before = mesh;
        mesh = snap_mesh(mesh, snap_points_, tols, lattice_.first_extended_bbox(),
                         res * rel_snap_dist_, aniso_);
        if (same(mesh, before)) {
          break;
        }
      }
      // Thin then smooth, twice: the smoother's flips reconfigure the mesh so a second thinning
      // pass can collapse redundant faces the first could not reach.
      for (auto pass = 0; pass < 2; pass++) {
        mesh = thin_snapped_mesh(mesh, snap_points_, tols, res, aniso_);
        mesh = smooth_snapped_mesh(mesh, snap_points_, tols, res, aniso_);
      }
    } else if (!mesh.is_empty()) {
      // No snapping: still smooth once. With no snap points the honor guard is inert, so this is a
      // pure bend-minimizing flip pass that aligns edges to curvature mesh-wide.
      mesh = smooth_snapped_mesh(mesh, snap_points_, VecX(), res, aniso_);
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
  double rel_snap_dist_{0.5};
  VecX rel_snap_tols_;
  int snap_max_passes_{8};
};

}  // namespace polatory::isosurface
