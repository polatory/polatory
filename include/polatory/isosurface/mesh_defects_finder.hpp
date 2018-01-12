// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>
#include <vector>

#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>

namespace polatory {
namespace isosurface {

class mesh_defects_finder {
  using halfedge = std::pair<vertex_index, vertex_index>;

  using face_index = int;
  using face_index_bool = std::pair<face_index, bool>;
  using face_index_bools = std::vector<face_index_bool>;

public:
  using edge = std::pair<vertex_index, vertex_index>;

  mesh_defects_finder(const std::vector<geometry::point3d>& vertices, const std::vector<face>& faces);

  std::vector<face> intersecting_faces() const;

  std::vector<edge> non_manifold_edges() const;

  std::vector<vertex_index> non_manifold_vertices() const;

private:
  face_index_bools::iterator halfedge_face(face_index_bools& fi_bools, halfedge he) const;  // NOLINT(runtime/references)

  bool line_triangle_intersects(vertex_index s1, vertex_index s2, const face& f) const;

  static halfedge opposite_halfedge(halfedge e);

  bool segment_crosses_the_plane(vertex_index s1, vertex_index s2, const face& f) const;

  halfedge vertex_incoming_halfedge(face_index fi, const vertex_index vi) const;

  halfedge vertex_outgoing_halfedge(face_index fi, const vertex_index vi) const;

  const std::vector<geometry::point3d>& vertices_;
  const std::vector<face>& faces_;
};

}  // namespace isosurface
}  // namespace polatory
