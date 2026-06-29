#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/clip.hpp>
#include <polatory/isosurface/cluster.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/rmt/lattice.hpp>
#include <polatory/isosurface/sign.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::isosurface {

class Isosurface {
  using Bbox3 = geometry::Bbox3;
  using Points3 = geometry::Points3;

 public:
  Isosurface(const Bbox3& bbox, double resolution)
      : Isosurface(bbox, resolution, Mat3::Identity()) {}

  Isosurface(const Bbox3& bbox, double resolution, const Mat3& aniso)
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

  Mesh generate_from_seed_points(const Points3& seed_points, FieldFunction& field_fn,
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

  void set_snap_points(const Points3& points, double relative_distance = 0.5,
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
    auto res = lattice_.resolution();

    auto mesh = lattice_.get_mesh();
    if (!mesh.is_empty()) {
      mesh = defeature(mesh, lattice_);

      for (auto pass = 0; pass < 2; pass++) {
        auto before = mesh.vertices().rows();
        mesh = cluster_vertices(mesh, lattice_, aniso_);
        if (pass > 0 && mesh.vertices().rows() == before) {
          break;
        }
        mesh = smooth_snapped_mesh(mesh, Points3(), VecX(), res, aniso_);
      }

      if (snap_points_.rows() != 0) {
        VecX tols = res * rel_snap_tols_;
        auto passes = tols.size() != 0 && tols.maxCoeff() > 0.0 ? snap_max_passes_ : 1;
        for (auto pass = 0; pass < passes; pass++) {
          Stats stats;
          mesh = snap_mesh(mesh, snap_points_, tols, lattice_.first_extended_bbox(),
                           res * rel_snap_dist_, aniso_, &stats);
          if (!stats.changed()) {
            break;
          }
        }

        for (auto pass = 0; pass < 2; pass++) {
          mesh = thin_snapped_mesh(mesh, snap_points_, tols, res, aniso_);
          mesh = smooth_snapped_mesh(mesh, snap_points_, tols, res, aniso_);
        }
      }

      mesh = clip(mesh, lattice_.bbox());
    }

    if (mesh.is_empty() &&
        lattice_.value_sign_at_arbitrary_point_within_bbox() == BinarySign::kNeg) {
      mesh = Mesh(EntireTag{});
    }

    lattice_.clear();

    return mesh;
  }

  rmt::Lattice lattice_;
  Mat3 aniso_;
  Points3 snap_points_;
  double rel_snap_dist_{0.5};
  VecX rel_snap_tols_;
  int snap_max_passes_{8};
};

}  // namespace polatory::isosurface
