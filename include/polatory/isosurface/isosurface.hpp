#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/clip.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt/lattice.hpp>
#include <polatory/isosurface/rmt/surface_generator.hpp>
#include <polatory/isosurface/surface.hpp>
#include <stdexcept>
#include <unordered_set>

namespace polatory::isosurface {

class isosurface {
 public:
  isosurface(const geometry::bbox3d& bbox, double resolution)
      : isosurface(bbox, resolution, geometry::matrix3d::Identity()) {}

  isosurface(const geometry::bbox3d& bbox, double resolution, const geometry::matrix3d& aniso)
      : rmt_lattice_(bbox, resolution, aniso) {
    if (!(aniso.determinant() > 0.0)) {
      throw std::invalid_argument("aniso must have a positive determinant");
    }
  }

  surface generate(field_function& field_fn, double isovalue = 0.0, int refine = 1) {
    if (refine < 0) {
      throw std::runtime_error("refine must be non-negative");
    }

    field_fn.set_evaluation_bbox(rmt_lattice_.extended_bbox());

    rmt_lattice_.add_all_nodes(field_fn, isovalue);
    rmt_lattice_.refine_vertices(field_fn, isovalue, refine);

    return generate_common();
  }

  surface generate_from_seed_points(const geometry::points3d& seed_points, field_function& field_fn,
                                    double isovalue = 0.0, int refine = 1) {
    if (refine < 0) {
      throw std::runtime_error("refine must be non-negative");
    }

    field_fn.set_evaluation_bbox(rmt_lattice_.extended_bbox());

    for (auto p : seed_points.rowwise()) {
      rmt_lattice_.add_cell_from_point(p);
    }
    rmt_lattice_.add_nodes_by_tracking(field_fn, isovalue);
    rmt_lattice_.refine_vertices(field_fn, isovalue, refine);

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

      auto num_unclustered = rmt_lattice_.uncluster_vertices(vertices_to_uncluster);
      if (num_unclustered == 0) {
        break;
      }
    }

    surface surf(rmt_lattice_.get_vertices(), rmt_surf.get_faces());
    surf.remove_unreferenced_vertices();

    if (surf.is_empty()) {
      if (rmt_lattice_.value_at_arbitrary_point() < 0.0) {
        surf = surface(entire_tag{});
      }
    } else {
      surf = clip_surface(surf, rmt_lattice_.bbox());
    }

    rmt_lattice_.clear();

    return surf;
  }

  rmt::lattice rmt_lattice_;
};

}  // namespace polatory::isosurface
