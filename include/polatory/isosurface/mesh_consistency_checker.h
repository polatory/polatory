#pragma once

#include <Eigen/Core>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::isosurface {

class mesh_consistency_checker {
  using vertices_type = geometry::points3d;
  using face_type = Eigen::Matrix<index_t, 1, 3>;
  using faces_type = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

 public:
  mesh_consistency_checker(const vertices_type& vertices, const faces_type& faces);

 private:
  geometry::point3d face_centroid(index_t fi) const {
    geometry::point3d a = vertices_.row(faces_(fi, 0));
    geometry::point3d b = vertices_.row(faces_(fi, 1));
    geometry::point3d c = vertices_.row(faces_(fi, 2));
    return (a + b + c) / 3.0;
  }

  geometry::vector3d face_normal(index_t fi) const {
    geometry::point3d a = vertices_.row(faces_(fi, 0));
    geometry::point3d b = vertices_.row(faces_(fi, 1));
    geometry::point3d c = vertices_.row(faces_(fi, 2));
    return (b - a).cross(c - a);
  }

  const vertices_type& vertices_;
  const faces_type& faces_;
};

}  // namespace polatory::isosurface
