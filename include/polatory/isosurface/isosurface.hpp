#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/clip.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice.hpp>
#include <polatory/isosurface/sign.hpp>
#include <polatory/isosurface/post_snap.hpp>
#include <polatory/isosurface/snap.hpp>
#include <polatory/types.hpp>
#include <array>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

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
  // relative_distance bounds how far a point may lie from the mesh to be snapped,
  // as a fraction of the mesh resolution. It must be in (0, 1].
  //
  // relative_tolerances, if non-empty, gives a per-point snapping tolerance as a fraction of
  // the mesh resolution -- the distance the surface may stay from the point. A point the
  // (partially snapped) mesh already passes within its tolerance of is skipped (snapping it
  // would barely move the surface and only over-subdivide the patch), and afterwards an inserted
  // vertex whose removal keeps the surface within its tolerance is dropped, so a densely sampled
  // polyline does not over-triangulate the surface. It must have one entry per point, each in
  // [0, relative_distance]; an empty vector means zero (snap every point in range, thin nothing).
  //
  // max_passes bounds how many times snapping is re-applied to recover points that lost contention
  // (see snap_mesh). It must be at least 1, and only matters when some tolerance is positive.
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

    // Snap the mesh to the points before clipping. Snapping is re-applied until a pass changes
    // nothing: a point that lost contention can snap to the finer mesh a later pass leaves behind.
    // A positive tolerance's slack is what lets that settle, so without one a single pass is used.
    // Restricting snapping to points whose projection stays inside first_extended_bbox keeps it
    // away from the mesh boundary, which lies further out (near second_extended_bbox); the clip
    // then makes the on-bbox boundary. Afterwards post_snap flattens the snapped region by edge
    // flips, leaving the base lattice elsewhere exactly as generated.
    if (snap_points_.rows() > 0 && !mesh.is_empty()) {
      auto res = lattice_.resolution();
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
      mesh = post_snap(mesh, res, aniso_, snapped_vertices(mesh));
    }

    mesh = clip(mesh, lattice_.bbox());

    if (mesh.is_empty() &&
        lattice_.value_sign_at_arbitrary_point_within_bbox() == BinarySign::kNeg) {
      mesh = Mesh(EntireTag{});
    }

    lattice_.clear();

    return mesh;
  }

  // Marks each vertex that coincides with a snap point, so the smoother can restrict its flips to
  // the snapped region. A captured point becomes a vertex emitted at its exact world position (the
  // snapper copies it through with no arithmetic), so an exact coordinate key matches it bit for
  // bit; an unsnapped vertex sits on the lattice and cannot collide.
  std::vector<bool> snapped_vertices(const Mesh& mesh) const {
    const auto& vertices = mesh.vertices();
    std::vector<bool> snapped(vertices.rows(), false);
    auto key = [](const auto& p) { return std::array<double, 3>{p.x(), p.y(), p.z()}; };
    struct Hash {
      std::size_t operator()(const std::array<double, 3>& k) const noexcept {
        std::hash<double> h;
        return h(k[0]) ^ (h(k[1]) << 1) ^ (h(k[2]) << 2);
      }
    };
    std::unordered_set<std::array<double, 3>, Hash> points;
    points.reserve(snap_points_.rows());
    for (Index i = 0; i < snap_points_.rows(); i++) {
      points.insert(key(snap_points_.row(i)));
    }
    for (Index v = 0; v < vertices.rows(); v++) {
      if (points.contains(key(vertices.row(v)))) {
        snapped[v] = true;
      }
    }
    return snapped;
  }

  rmt::Lattice lattice_;
  Mat3 aniso_;
  geometry::Points3 snap_points_;
  double rel_snap_dist_{0.5};
  VecX rel_snap_tols_;
  int snap_max_passes_{8};
};

}  // namespace polatory::isosurface
