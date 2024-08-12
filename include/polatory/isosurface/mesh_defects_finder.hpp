#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

class mesh_defects_finder {
  using Face = face;
  using Faces = faces;
  using Mesh = mesh;
  using Points = geometry::points3d;

 public:
  explicit mesh_defects_finder(const Mesh& mesh);

  std::vector<index_t> intersecting_faces() const;

  std::vector<index_t> singular_vertices() const;

 private:
  bool edge_face_intersect(index_t vi, index_t vj, index_t fi) const;

  index_t next_vertex(index_t fi, index_t vi) const;

  index_t prev_vertex(index_t fi, index_t vi) const;

  const Points& vertices_;
  const Faces& faces_;
  std::vector<std::vector<index_t>> vf_map_;
};

}  // namespace polatory::isosurface
