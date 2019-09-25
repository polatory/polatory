// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/rbf_field_function.hpp>
#include <polatory/isosurface/surface.hpp>
#include <polatory/isosurface/types.hpp>

namespace polatory {
namespace isosurface {

class isosurface_2d {
public:
  isosurface_2d(const geometry::bbox3d& bbox, double resolution)
    : bbox_(bbox)
    , resolution_(resolution) {
  }

  surface generate(field_function& field_fn) const {  // NOLINT(runtime/references)
    field_fn.set_evaluation_bbox(bbox_);

    auto xmin = resolution_ * (std::floor(bbox_.min()(0) / resolution_) - 1.0);
    auto ymin = resolution_ * (std::floor(bbox_.min()(1) / resolution_) - 1.0);
    auto xmax = resolution_ * (std::ceil(bbox_.max()(0) / resolution_) + 1.0);
    auto ymax = resolution_ * (std::ceil(bbox_.max()(1) / resolution_) + 1.0);
    auto nx = static_cast<vertex_index>(std::round((xmax - xmin) / resolution_)) + 1;
    auto ny = static_cast<vertex_index>(std::round((ymax - ymin) / resolution_)) + 1;

    geometry::points3d points(ny * nx, 3);
    for (vertex_index iy = 0; iy < ny; iy++) {
      for (vertex_index ix = 0; ix < nx; ix++) {
        points(iy * nx + ix, 0) = xmin + ix * resolution_;
        points(iy * nx + ix, 1) = ymin + iy * resolution_;
        points(iy * nx + ix, 2) = 0.0;
      }
    }
    points.rightCols(1) = field_fn(points);

    std::vector<geometry::point3d> vertices;
    for (vertex_index i = 0; i < points.rows(); i++) {
      vertices.emplace_back(points.row(i));
    }

    std::vector<face> faces;
    for (vertex_index iy = 0; iy < ny - 1; iy++) {
      for (vertex_index ix = 0; ix < nx - 1; ix++) {
        auto i0 = iy * nx + ix;
        auto i1 = iy * nx + (ix + 1);
        auto i2 = (iy + 1) * nx + ix;
        auto i3 = (iy + 1) * nx + (ix + 1);
        faces.push_back({ i0, i1, i3 });
        faces.push_back({ i0, i3, i2 });
      }
    }

    return { vertices, faces };
  }

private:
  const geometry::bbox3d bbox_;
  const double resolution_;
};

}  // namespace isosurface
}  // namespace polatory
