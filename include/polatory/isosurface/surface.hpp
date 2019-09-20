// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <polatory/common/exception.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/numeric/roundtrip_string.hpp>

namespace polatory {
namespace isosurface {

class surface {
public:
  using vertices_type = std::vector<geometry::point3d>;
  using faces_type = std::vector<face>;

  surface(vertices_type vertices, faces_type faces)
    : vertices_(std::move(vertices))
    , faces_(std::move(faces)) {
  }

  bool export_obj(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs)
      throw common::io_error("Could not open file '" + filename + "'.");

    for (auto& v : vertices_) {
      ofs << "v "
          << numeric::to_string(v[0]) << ' '
          << numeric::to_string(v[1]) << ' '
          << numeric::to_string(v[2]) << '\n';
    }

    for (auto& f : faces_) {
      ofs << "f " << f[0] + 1 << ' ' << f[1] + 1 << ' ' << f[2] + 1 << '\n';
    }

    return true;
  }

  const faces_type& faces() const {
    return faces_;
  }

  const vertices_type& vertices() const {
    return vertices_;
  }

private:
  const vertices_type vertices_;
  const faces_type faces_;
};

}  // namespace isosurface
}  // namespace polatory
