#pragma once

#include <Eigen/Core>
#include <fstream>
#include <polatory/geometry/point3d.hpp>
#include <polatory/numeric/conv.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory::isosurface {

struct entire_tag {};

class surface {
 public:
  using vertices_type = geometry::points3d;
  using face_type = Eigen::Matrix<index_t, 1, 3>;
  using faces_type = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

  surface(const vertices_type& vertices, const faces_type& faces)
      : vertices_(vertices.rows(), 3), faces_(faces.rows(), 3) {
    std::vector<index_t> vi_map(vertices.rows(), -1);
    face_type new_face;

    auto v_it = vertices_.rowwise().begin();
    auto f_it = faces_.rowwise().begin();
    index_t n_vertices = 0;
    index_t n_faces = 0;

    for (auto face : faces.rowwise()) {
      for (auto i = 0; i < 3; i++) {
        auto& vi = vi_map.at(face(i));
        if (vi == -1) {
          vi = n_vertices;
          *v_it++ = vertices.row(face(i));
          n_vertices++;
        }
        new_face(i) = vi;
      }
      *f_it++ = new_face;
      n_faces++;
    }

    vertices_.conservativeResize(n_vertices, 3);
    faces_.conservativeResize(n_faces, 3);
  }

  explicit surface(entire_tag /*tag*/) : entire_(true) {}

  void export_obj(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs) {
      throw std::runtime_error("Failed to open file '" + filename + "'.");
    }

    if (faces_.rows() == 0) {
      if (entire_) {
        ofs << "# entire\n";
      } else {
        ofs << "# empty\n";
      }
    }

    for (auto v : vertices_.rowwise()) {
      ofs << "v " << numeric::to_string(v(0)) << ' ' << numeric::to_string(v(1)) << ' '
          << numeric::to_string(v(2)) << '\n';
    }

    for (auto f : faces_.rowwise()) {
      ofs << "f " << f(0) + 1 << ' ' << f(1) + 1 << ' ' << f(2) + 1 << '\n';
    }
  }

  const faces_type& faces() const { return faces_; }

  bool is_empty() const { return faces_.rows() == 0 && !entire_; }

  bool is_entire() const { return entire_; }

  const vertices_type& vertices() const { return vertices_; }

 private:
  vertices_type vertices_;
  faces_type faces_;
  bool entire_{};
};

}  // namespace polatory::isosurface
