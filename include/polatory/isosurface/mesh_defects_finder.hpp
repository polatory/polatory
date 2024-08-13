#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/mesh.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

class MeshDefectsFinder {
  using Points = geometry::Points3;

 public:
  explicit MeshDefectsFinder(const Mesh& mesh);

  std::vector<Index> intersecting_faces() const;

  std::vector<Index> singular_vertices() const;

 private:
  bool edge_face_intersect(Index vi, Index vj, Index fi) const;

  Index next_vertex(Index fi, Index vi) const;

  Index prev_vertex(Index fi, Index vi) const;

  const Points& vertices_;
  const Faces& faces_;
  std::vector<std::vector<Index>> vf_map_;
};

}  // namespace polatory::isosurface
