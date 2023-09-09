#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <vector>

namespace polatory::isosurface {

class mesh_defects_finder {
  using vertices_type = geometry::points3d;
  using face_type = Eigen::Matrix<index_t, 1, 3>;
  using faces_type = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

 public:
  mesh_defects_finder(const vertices_type& vertices, const faces_type& faces);

  std::unordered_set<index_t> intersecting_faces() const;

  std::unordered_set<index_t> singular_vertices() const;

 private:
  bool line_triangle_intersect(index_t vi, index_t vj, index_t fi) const;

  bool segment_plane_intersect(index_t vi, index_t vj, index_t fi) const;

  bool segment_triangle_intersect(index_t vi, index_t vj, index_t fi) const;

  index_t next_vertex(index_t fi, index_t vi) const;

  index_t prev_vertex(index_t fi, index_t vi) const;

  const vertices_type& vertices_;
  const faces_type& faces_;
  std::vector<std::vector<index_t>> vf_map_;
};

}  // namespace polatory::isosurface
