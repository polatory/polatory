// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <set>
#include <utility>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh_defects_finder.hpp>
#include <polatory/isosurface/rmt_lattice.hpp>
#include <polatory/isosurface/rmt_surface.hpp>
#include <polatory/isosurface/surface.hpp>

namespace polatory {
namespace isosurface {

class isosurface {
public:
  isosurface(const geometry::bbox3d& bbox, double resolution)
    : rmt_lattice_(bbox, resolution) {
  }

  surface generate(field_function& field_fn, double isovalue = 0.0) {  // NOLINT(runtime/references)
    field_fn.set_evaluation_bbox(rmt_lattice_.extended_bbox());

    rmt_lattice_.add_all_nodes(field_fn, isovalue);

    return generate_common();
  }

  surface generate_from_seed_points(const geometry::points3d& seed_points, field_function& field_fn, double isovalue = 0.0) {  // NOLINT(runtime/references)
    field_fn.set_evaluation_bbox(rmt_lattice_.extended_bbox());

    for (auto p : common::row_range(seed_points)) {
      rmt_lattice_.add_cell_contains_point(p);
    }
    rmt_lattice_.add_nodes_by_tracking(field_fn, isovalue);

    return generate_common();
  }

private:
  surface generate_common() {
    rmt_lattice_.cluster_vertices();

    rmt_surface rmt_surf(rmt_lattice_);
    rmt_surf.generate_surface();

    mesh_defects_finder defects(rmt_lattice_.get_vertices(), rmt_surf.get_faces());

    auto vertices = defects.singular_vertices();
    auto faces = defects.intersecting_faces();

    std::set<vertex_index> vertices_to_uncluster;
    for (auto vertex : vertices) {
      vertices_to_uncluster.insert(vertex);
    }
    for (auto face : faces) {
      vertices_to_uncluster.insert(face[0]);
      vertices_to_uncluster.insert(face[1]);
      vertices_to_uncluster.insert(face[2]);
    }

    rmt_lattice_.uncluster_vertices(vertices_to_uncluster.begin(), vertices_to_uncluster.end());
    rmt_lattice_.remove_unreferenced_vertices();
    rmt_surf.generate_surface();

    surface surf(std::move(rmt_lattice_.get_vertices()), std::move(rmt_surf.get_faces()));
    rmt_lattice_.clear();

    return surf;
  }

  rmt_lattice rmt_lattice_;
};

}  // namespace isosurface
}  // namespace polatory
