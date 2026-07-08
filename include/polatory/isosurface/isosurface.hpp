#pragma once

#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/clip.hpp>
#include <polatory/isosurface/cluster.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/refine.hpp>
#include <polatory/isosurface/relax.hpp>
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

  Mesh generate(FieldFunction& field_fn, double isovalue = 0.0, bool refine = true) {
    field_fn.set_evaluation_bbox(lattice_.second_extended_bbox());

    lattice_.add_all_nodes(field_fn, isovalue);

    return generate_common(field_fn, isovalue, refine);
  }

  Mesh generate_from_seed_points(const Points3& seed_points, FieldFunction& field_fn,
                                 double isovalue = 0.0, bool refine = true) {
    if (seed_points.rows() == 0) {
      throw std::runtime_error("seed points must not be empty");
    }

    field_fn.set_evaluation_bbox(lattice_.second_extended_bbox());

    lattice_.add_nodes_from_seed_points(seed_points, field_fn, isovalue);

    return generate_common(field_fn, isovalue, refine);
  }

  void set_snap_points(const Points3& points, const VecX& relative_tolerances = VecX()) {
    if (relative_tolerances.size() != 0) {
      if (relative_tolerances.size() != points.rows()) {
        throw std::invalid_argument("snap relative tolerances must have one entry per point");
      }
      if (!(relative_tolerances.minCoeff() >= 0.0 && relative_tolerances.maxCoeff() <= 1.0)) {
        throw std::invalid_argument("snap relative tolerances must be in [0, 1]");
      }
    }

    snap_points_ = points;
    rel_snap_tols_ = relative_tolerances;
  }

 private:
  Mesh generate_common(const FieldFunction& field_fn, double isovalue, bool refine) {
    auto res = lattice_.resolution();
    // The lattice's natural edge is ~1.2x its resolution; target the remesh at 1.2 * res so the
    // output lands on the desired edge length rather than the lattice's coarser natural edge. The
    // caller builds the lattice 1.2x finer than the edge length it wants.
    auto rres = 1.2 * res;

    auto mesh = lattice_.get_mesh();
    if (!mesh.is_empty()) {
      for (auto pass = 0; pass < 2; pass++) {
        mesh = cluster_vertices(mesh, lattice_, aniso_);
        if (refine && pass == 1) {
          mesh = refine_vertices(mesh, field_fn, isovalue, lattice_.bbox(), rres, aniso_);
        }
        mesh = smooth_snapped_mesh(mesh, Points3(), VecX(), rres, aniso_);
      }

      VecX tols = rres * rel_snap_tols_;  // empty unless snap points were set
      if (snap_points_.rows() != 0) {
        // Relax, snap, and smooth interleave so a later snap can reclaim points that a relaxation
        // or within-pass contention left dishonored. Edge optimization stays out: a feature snaps
        // gradually (it grows from where it is reachable), and honors_ok protects only points
        // already honored, so a collapse here would shrink a feature still forming.
        std::vector<std::size_t> mesh_hashes;
        for (auto iter = 0; iter < 20; iter++) {
          Stats stats;
          mesh = snap_mesh(mesh, snap_points_, tols, rres, aniso_, &stats);
          mesh = smooth_snapped_mesh(mesh, snap_points_, tols, rres, aniso_);

          // Some points are unreachable (sub-resolution contention); a deterministic pass that
          // repeats an earlier mesh has reached a fixpoint or a cycle, so stop either way.
          auto hash = hash_mesh(mesh);
          if (std::ranges::find(mesh_hashes, hash) != mesh_hashes.end()) {
            break;
          }

          mesh_hashes.push_back(hash);
        }

        // Snapping has settled: every reachable point is now honored, so honors_ok fully protects
        // the features and no feature is still forming for a collapse to fight. Remesh toward
        // uniform edges: optimize edge lengths (split long / collapse short), flip toward valence,
        // then relax the vertices. Relaxation slides snap points freely but re-projects them onto
        // the level set, which holds them on the features; it equalizes the triangles the split and
        // collapse leave, so a later round's collapse frees a sliver its guards had to defer.
        for (auto pass = 0; pass < 4; pass++) {
          mesh = optimize_edges(mesh, snap_points_, tols, rres, aniso_);
          mesh = smooth_snapped_mesh(mesh, snap_points_, tols, rres, aniso_);
          mesh = relax_vertices(mesh, field_fn, isovalue, rres, aniso_, 5, lattice_.bbox());
        }

        // Relaxation slides the honored vertices tangentially along the features; a final snap
        // re-places any point it pushed outside tolerance. move_nearby_vertices is off here: a
        // point still honored within tolerance keeps its relaxed position rather than having a
        // vertex pulled exactly onto it, so the uniform edges survive.
        mesh = snap_mesh(mesh, snap_points_, tols, rres, aniso_, nullptr, false);
      } else {
        for (auto pass = 0; pass < 4; pass++) {
          mesh = optimize_edges(mesh, snap_points_, tols, rres, aniso_);
          mesh = smooth_snapped_mesh(mesh, snap_points_, tols, rres, aniso_);
          mesh = relax_vertices(mesh, field_fn, isovalue, rres, aniso_, 5, lattice_.bbox());
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

  static std::size_t hash_mesh(const Mesh& m) {
    std::size_t seed = 0;
    const auto& v = m.vertices();
    const auto& f = m.faces();
    boost::hash_combine(seed, boost::hash_range(v.data(), v.data() + v.size()));
    boost::hash_combine(seed, boost::hash_range(f.data(), f.data() + f.size()));
    return seed;
  }

  rmt::Lattice lattice_;
  Mat3 aniso_;
  Points3 snap_points_;
  VecX rel_snap_tols_;
};

}  // namespace polatory::isosurface
