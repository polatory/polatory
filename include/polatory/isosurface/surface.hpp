#pragma once

#include <Eigen/Core>
#include <format>
#include <fstream>
#include <polatory/geometry/point3d.hpp>
#include <polatory/numeric/conv.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace polatory::isosurface {

struct entire_tag {};

class surface {
  using Face = Eigen::Matrix<index_t, 1, 3>;
  using Faces = Eigen::Matrix<index_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using Points = geometry::points3d;

 public:
  surface() = default;

  surface(Points vertices, Faces faces)
      : vertices_(std::move(vertices)), faces_(std::move(faces)) {}

  void remove_unreferenced_vertices() {
    Points new_vertices(vertices_.rows(), 3);

    std::vector<index_t> v_map(vertices_.rows(), -1);

    index_t n_vertices = 0;
    for (auto face : faces_.rowwise()) {
      Face new_face;
      for (auto i = 0; i < 3; i++) {
        auto old_v = face(i);
        auto& new_v = v_map.at(old_v);
        if (new_v == -1) {
          new_v = n_vertices;
          new_vertices.row(new_v) = vertices_.row(old_v);
          n_vertices++;
        }
        new_face(i) = new_v;
      }
      face = new_face;
    }

    new_vertices.conservativeResize(n_vertices, 3);
    vertices_ = std::move(new_vertices);
  }

  explicit surface(entire_tag /*tag*/) : entire_(true) {}

  void export_obj(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs) {
      throw std::runtime_error(std::format("cannot open file '{}'", filename));
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

  const Faces& faces() const { return faces_; }

  bool is_empty() const { return faces_.rows() == 0 && !entire_; }

  bool is_entire() const { return entire_; }

  const Points& vertices() const { return vertices_; }

 private:
  Points vertices_;
  Faces faces_;
  bool entire_{};
};

}  // namespace polatory::isosurface
