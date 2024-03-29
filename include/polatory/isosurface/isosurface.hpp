#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice.hpp>
#include <polatory/isosurface/rmt/surface_generator.hpp>
#include <polatory/isosurface/surface.hpp>
#include <unordered_set>

namespace polatory::isosurface {

class isosurface {
 public:
  isosurface(const geometry::bbox3d& bbox, double resolution) : rmt_lattice_(bbox, resolution) {}

  surface generate(field_function& field_fn, double isovalue = 0.0, bool refine = true) {
    field_fn.set_evaluation_bbox(rmt_lattice_.extended_bbox());

    rmt_lattice_.add_all_nodes(field_fn, isovalue);
    if (refine) {
      rmt_lattice_.refine_vertices(field_fn, isovalue);
    }

    return generate_common();
  }

  surface generate_from_seed_points(const geometry::points3d& seed_points, field_function& field_fn,
                                    double isovalue = 0.0, bool refine = true) {
    field_fn.set_evaluation_bbox(rmt_lattice_.extended_bbox());

    for (auto p : seed_points.rowwise()) {
      rmt_lattice_.add_cell_contains_point(p);
    }
    rmt_lattice_.add_nodes_by_tracking(field_fn, isovalue);
    if (refine) {
      rmt_lattice_.refine_vertices(field_fn, isovalue);
    }

    return generate_common();
  }

 private:
  surface generate_common() {
    rmt_lattice_.cluster_vertices();

    rmt::surface_generator rmt_surf(rmt_lattice_);
    rmt_surf.generate_surface();

    // Unclustering non-manifold vertices may require a few iterations.
    while (true) {
      auto vertices = rmt_lattice_.get_vertices();
      auto faces = rmt_surf.get_faces();
      mesh_defects_finder defects(vertices, faces);

      auto vis = defects.singular_vertices();
      auto fis = defects.intersecting_faces();

      std::unordered_set<vertex_index> vertices_to_uncluster;
      for (auto vi : vis) {
        vertices_to_uncluster.insert(vi);
      }
      for (auto fi : fis) {
        auto f = faces.row(fi);
        vertices_to_uncluster.insert(f(0));
        vertices_to_uncluster.insert(f(1));
        vertices_to_uncluster.insert(f(2));
      }

      rmt_lattice_.uncluster_vertices(vertices_to_uncluster);

      // fis.empty() is not checked as it is not possible to eliminate all self intersections.
      if (vis.empty()) {
        break;
      }
    }

    surface surf(rmt_lattice_.get_vertices(), rmt_surf.get_faces());

    if (surf.is_empty()) {
      if (rmt_lattice_.value_at_arbitrary_point() < 0.0) {
        surf = surface(entire_tag{});
      }
    }

    rmt_lattice_.clear();

    return surf;
  }

  rmt::lattice rmt_lattice_;
};

}  // namespace polatory::isosurface
