// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <polatory/common/pi.hpp>
#include <polatory/geometry/affine_transform3d.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>

namespace polatory {
namespace isosurface {

namespace {

// RotationMatrix[-Pi/2, {0, 0, 1}].RotationMatrix[-Pi/4, {0, 1, 0}]
geometry::affine_transform3d rotation() {
  return geometry::affine_transform3d::roll_pitch_yaw({ -common::pi<double> / 2.0, 0.0, -common::pi<double> / 4.0 });
}

// Primitive vectors of body-centered cubic.
std::array<geometry::vector3d, 3> PrimitiveVectors
  {
    rotation().transform_vector({ +1., +1., -1. }),
    rotation().transform_vector({ +1., -1., +1. }),
    rotation().transform_vector({ -1., +1., +1. })
  };

// Reciprocal primitive vectors of body-centered cubic.
std::array<geometry::vector3d, 3> ReciprocalPrimitiveVectors
  {
    rotation().transform_vector({ 1. / 2., 1. / 2., 0. }),
    rotation().transform_vector({ 1. / 2., 0., 1. / 2. }),
    rotation().transform_vector({ 0., 1. / 2., 1. / 2. })
  };

} // namespace


class rmt_primitive_lattice {
protected:
  // Original bbox
  geometry::bbox3d bbox;

  // Extended bbox
  geometry::bbox3d ext_bbox;

  // Lattice constant
  double lc;
  // Reciprocal lattice constant
  double rlc;

  // Primitive vectors scaled by `lc`.
  geometry::vector3d a0;
  geometry::vector3d a1;
  geometry::vector3d a2;

  // Reciprocal primitive vectors scaled by `rlc`.
  geometry::vector3d b0;
  geometry::vector3d b1;
  geometry::vector3d b2;

  cell_vector cell_min;
  cell_vector cell_max;
  const unsigned int shift1 = 21;
  const unsigned int shift2 = 42;
  const cell_index mask = (cell_index{ 1 } << 21) - 1;

private:
  // TODO: Make sure we have some additional margins so that
  // neighbor_cell_index() will never return unintended cell index
  // when an index of a boundary cell is passed.
  void initialize() {
    a0 = lc * PrimitiveVectors[0];
    a1 = lc * PrimitiveVectors[1];
    a2 = lc * PrimitiveVectors[2];

    b0 = rlc * ReciprocalPrimitiveVectors[0];
    b1 = rlc * ReciprocalPrimitiveVectors[1];
    b2 = rlc * ReciprocalPrimitiveVectors[2];

    auto sqrt2 = std::sqrt(2.0);
    geometry::vector3d cell_hull = lc * geometry::vector3d(3.0, 2.0 * sqrt2, sqrt2);

    // Extend each side of bbox by a primitive cell
    // to ensure all required nodes are inside the extended bbox.
    geometry::vector3d ext = cell_hull * (1.0 + std::pow(2.0, -5.0));
    ext_bbox = geometry::bbox3d(bbox.min() - ext, bbox.max() + ext);

    geometry::points3d ext_bbox_vertices(8, 3);
    ext_bbox_vertices <<
      ext_bbox.min()(0), ext_bbox.min()(1), ext_bbox.min()(2),
      ext_bbox.max()(0), ext_bbox.min()(1), ext_bbox.min()(2),
      ext_bbox.min()(0), ext_bbox.max()(1), ext_bbox.min()(2),
      ext_bbox.min()(0), ext_bbox.min()(1), ext_bbox.max()(2),
      ext_bbox.min()(0), ext_bbox.max()(1), ext_bbox.max()(2),
      ext_bbox.max()(0), ext_bbox.min()(1), ext_bbox.max()(2),
      ext_bbox.max()(0), ext_bbox.max()(1), ext_bbox.min()(2),
      ext_bbox.max()(0), ext_bbox.max()(1), ext_bbox.max()(2);

    geometry::vectors3d cell_vecsd(8, 3);

    for (size_t i = 0; i < 8; i++) {
      auto v = ext_bbox_vertices.row(i);
      cell_vecsd.row(i) = geometry::vector3d(v.dot(b0),  v.dot(b1),  v.dot(b2));
    }

    geometry::vector3d cell_mind = cell_vecsd.colwise().minCoeff();
    geometry::vector3d cell_maxd = cell_vecsd.colwise().maxCoeff();

    cell_min = cell_vector(
      static_cast<int>(std::floor(cell_mind(0))),
      static_cast<int>(std::floor(cell_mind(1))),
      static_cast<int>(std::floor(cell_mind(2)))
    );
    cell_max = cell_vector(
      static_cast<int>(std::ceil(cell_maxd(0))),
      static_cast<int>(std::ceil(cell_maxd(1))),
      static_cast<int>(std::ceil(cell_maxd(2)))
    );

    if (static_cast<cell_index>(cell_max(0) - cell_min(0)) > mask ||
        static_cast<cell_index>(cell_max(1) - cell_min(1)) > mask ||
        static_cast<cell_index>(cell_max(2) - cell_min(2)) > mask)
      throw std::range_error("Bounds are too large or resolution is too small.");
  }

public:
  rmt_primitive_lattice(const geometry::bbox3d& bbox, double resolution)
    : bbox(bbox)
    , lc(resolution / std::sqrt(2.0))
    , rlc(std::sqrt(2.0) / resolution) {
    initialize();
  }

  cell_index cell_contains_point(const geometry::point3d& p) const {
    auto m0 = static_cast<int>(std::floor(p.dot(b0)));
    auto m1 = static_cast<int>(std::floor(p.dot(b1)));
    auto m2 = static_cast<int>(std::floor(p.dot(b2)));

    cell_index offset2 = static_cast<cell_index>(m2 - cell_min(2)) << shift2;
    cell_index offset21 = offset2 | (static_cast<cell_index>(m1 - cell_min(1)) << shift1);
    return offset21 | static_cast<cell_index>(m0 - cell_min(0));
  }

  cell_vector cell_vector_from_index(cell_index cell_idx) const {
    int m0 = static_cast<int>(cell_idx & mask) + cell_min(0);
    int m1 = static_cast<int>((cell_idx >> shift1) & mask) + cell_min(1);
    int m2 = static_cast<int>((cell_idx >> shift2) & mask) + cell_min(2);

    return cell_vector(m0, m1, m2);
  }

  bool is_inside_bounds(const geometry::point3d& point) const {
    return
      point(0) >= ext_bbox.min()(0) && point(0) <= ext_bbox.max()(0) &&
      point(1) >= ext_bbox.min()(1) && point(1) <= ext_bbox.max()(1) &&
      point(2) >= ext_bbox.min()(2) && point(2) <= ext_bbox.max()(2);
  }

  geometry::bbox3d node_bounds() const {
    return ext_bbox;
  }

  geometry::point3d point_from_cell_vector(const cell_vector& cv) const {
    return cv(0) * a0 + cv(1) * a1 + cv(2) * a2;
  }
};

} // namespace isosurface
} // namespace polatory
