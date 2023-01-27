#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>
#include <set>
#include <vector>

namespace polatory {
namespace isosurface {

class mesh_defects_finder {
 public:
  mesh_defects_finder(const std::vector<geometry::point3d>& vertices,
                      const std::vector<face>& faces);

  std::set<face> intersecting_faces() const;

  std::set<vertex_index> singular_vertices() const;

 private:
  bool line_triangle_intersect(vertex_index s1, vertex_index s2, const face& f) const;

  bool segment_plane_intersect(vertex_index s1, vertex_index s2, const face& f) const;

  bool segment_triangle_intersect(vertex_index s1, vertex_index s2, const face& f) const;

  const std::vector<geometry::point3d>& vertices_;
  std::vector<std::vector<face>> vf_map_;
};

}  // namespace isosurface
}  // namespace polatory
