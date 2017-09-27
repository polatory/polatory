// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Core>

#include "../geometry/bbox3.hpp"
#include "mesh_defects_finder.hpp"
#include "rmt_lattice.hpp"
#include "rmt_surface.hpp"

namespace polatory {
namespace isosurface {

class isosurface {
  rmt_lattice lattice;
  rmt_surface surface;

  void generate_common() {
    lattice.cluster_vertices();
    surface.generate_surface();

    mesh_defects_finder defects(lattice.get_vertices(), surface.get_faces());

    auto edges = defects.non_manifold_edges();
    auto vertices = defects.non_manifold_vertices();
    auto faces = defects.intersecting_faces();

    std::set<vertex_index> vertices_to_uncluster;
    for (auto edge : edges) {
      vertices_to_uncluster.insert(edge.first);
      vertices_to_uncluster.insert(edge.second);
    }
    for (auto vertex : vertices) {
      vertices_to_uncluster.insert(vertex);
    }
    for (auto face : faces) {
      vertices_to_uncluster.insert(face[0]);
      vertices_to_uncluster.insert(face[1]);
      vertices_to_uncluster.insert(face[2]);
    }

    lattice.uncluster_vertices(vertices_to_uncluster.begin(), vertices_to_uncluster.end());
    lattice.remove_unreferenced_vertices();
    surface.generate_surface();
  }

public:
  isosurface(const geometry::bbox3d& bbox, double resolution)
    : lattice(bbox, resolution)
    , surface(lattice) {
  }

  geometry::bbox3d evaluation_bbox() const {
    return lattice.node_bounds();
  }

  const std::vector<face>& faces() const {
    return surface.get_faces();
  }

  void generate(const field_function& field_func, double isovalue = 0.0) {
    lattice.clear();

    lattice.add_all_nodes(field_func, isovalue);

    generate_common();
  }

  template <typename Container>
  void
  generate_from_seed_points(const Container& seed_points, const field_function& field_func, double isovalue = 0.0) {
    lattice.clear();

    for (auto& p : seed_points) {
      lattice.add_cell_contains_point(p);
    }
    lattice.add_nodes_by_tracking(field_func, isovalue);

    generate_common();
  }

  const std::vector<Eigen::Vector3d>& vertices() const {
    return lattice.get_vertices();
  }
};

} // namespace isosurface
} // namespace polatory
