#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <vector>

namespace polatory::isosurface {

class mesh_defects_finder {
  using Face = Eigen::Matrix<index_t, 1, 3>;
  using Faces = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using Points = geometry::points3d;

 public:
  mesh_defects_finder(const Points& vertices, const Faces& faces);

  std::unordered_set<index_t> intersecting_faces() const;

  std::unordered_set<index_t> singular_vertices() const;

 private:
  bool edge_face_intersect(index_t vi, index_t vj, index_t fi) const;

  index_t next_vertex(index_t fi, index_t vi) const;

  index_t prev_vertex(index_t fi, index_t vi) const;

  const Points& vertices_;
  const Faces& faces_;
  std::vector<std::vector<index_t>> vf_map_;
};

}  // namespace polatory::isosurface
