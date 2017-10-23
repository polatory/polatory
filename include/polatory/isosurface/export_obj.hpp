// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <fstream>
#include <string>

#include "isosurface.hpp"
#include "polatory/numeric/to_string.hpp"

namespace polatory {
namespace isosurface {

inline bool export_obj(std::string filename, const isosurface& isosurf) {
  std::ofstream ofs(filename);
  if (!ofs) return false;

  for (auto& v : isosurf.vertices()) {
    ofs << "v "
        << numeric::to_string(v[0]) << ' '
        << numeric::to_string(v[1]) << ' '
        << numeric::to_string(v[2]) << '\n';
  }

  for (auto& f : isosurf.faces()) {
    ofs << "f " << f[0] + 1 << ' ' << f[1] + 1 << ' ' << f[2] + 1 << '\n';
  }

  return true;
}

} // namespace isosurface
} // namespace polatory
