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
  MeshDefectsFinder(const Mesh& mesh, double resolution);

  std::vector<Index> intersecting_faces() const;

  std::vector<Index> singular_vertices() const;

 private:
  bool intersect(Index fi, Index fj) const;

  Index next_vertex(Index fi, Index vi) const;

  Index prev_vertex(Index fi, Index vi) const;

  const Points& vertices_;
  const Faces& faces_;
  double resolution_;
  Index nv_;
  Index nf_;
  std::vector<std::vector<Index>> vf_;
};

}  // namespace polatory::isosurface
