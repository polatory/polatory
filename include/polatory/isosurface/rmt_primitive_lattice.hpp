// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include <polatory/common/pi.hpp>
#include <polatory/geometry/affine_transformation3d.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>

namespace polatory {
namespace isosurface {

namespace detail {

class lattice_vectors : public std::array<geometry::vector3d, 3> {
  using base = std::array<geometry::vector3d, 3>;

public:
  lattice_vectors() noexcept;
};

class dusl_lattice_vectors : public std::array<geometry::vector3d, 3> {
  using base = std::array<geometry::vector3d, 3>;

public:
  dusl_lattice_vectors() noexcept;
};

}  // namespace detail

// RotationMatrix[-Pi/2, {0, 0, 1}].RotationMatrix[-Pi/4, {0, 1, 0}]
inline
geometry::affine_transformation3d rotation() {
  return geometry::affine_transformation3d::roll_pitch_yaw({ -common::pi<double>() / 2.0, 0.0, -common::pi<double>() / 4.0 });
}

// Primitive vectors of body-centered cubic.
extern const detail::lattice_vectors LatticeVectors;

// Reciprocal primitive vectors of body-centered cubic.
extern const detail::dusl_lattice_vectors DualLatticeVectors;

class rmt_primitive_lattice {
public:
  rmt_primitive_lattice(const geometry::bbox3d& bbox, double resolution)
    : a0_(resolution * LatticeVectors[0])
    , a1_(resolution * LatticeVectors[1])
    , a2_(resolution * LatticeVectors[2])
    , b0_(DualLatticeVectors[0] / resolution)
    , b1_(DualLatticeVectors[1] / resolution)
    , b2_(DualLatticeVectors[2] / resolution) {
    // Size of bbox of a single cell.
    geometry::vector3d cell_hull = resolution * geometry::vector3d(3.0 / std::sqrt(2.0), 2.0, 1.0);

    // Extend each side of bbox by a primitive cell
    // to ensure all required nodes are inside the extended bbox.
    geometry::vector3d ext = cell_hull * (1.0 + std::pow(2.0, -5.0));
    ext_bbox_ = geometry::bbox3d(bbox.min() - ext, bbox.max() + ext);

    geometry::points3d ext_bbox_vertices(8, 3);
    ext_bbox_vertices <<
      ext_bbox_.min()(0), ext_bbox_.min()(1), ext_bbox_.min()(2),
      ext_bbox_.max()(0), ext_bbox_.min()(1), ext_bbox_.min()(2),
      ext_bbox_.min()(0), ext_bbox_.max()(1), ext_bbox_.min()(2),
      ext_bbox_.min()(0), ext_bbox_.min()(1), ext_bbox_.max()(2),
      ext_bbox_.min()(0), ext_bbox_.max()(1), ext_bbox_.max()(2),
      ext_bbox_.max()(0), ext_bbox_.min()(1), ext_bbox_.max()(2),
      ext_bbox_.max()(0), ext_bbox_.max()(1), ext_bbox_.min()(2),
      ext_bbox_.max()(0), ext_bbox_.max()(1), ext_bbox_.max()(2);

    geometry::vectors3d cv_d(8, 3);
    for (auto i = 0; i < 8; i++) {
      cv_d.row(i) = cell_vector_d(ext_bbox_vertices.row(i));
    }

    geometry::vector3d cv_d_min = cv_d.colwise().minCoeff();
    geometry::vector3d cv_d_max = cv_d.colwise().maxCoeff();

    cv_min = cell_vector(
      static_cast<int>(std::floor(cv_d_min(0))),
      static_cast<int>(std::floor(cv_d_min(1))),
      static_cast<int>(std::floor(cv_d_min(2)))
    );
    cv_max = cell_vector(
      static_cast<int>(std::floor(cv_d_max(0))) + 1,
      static_cast<int>(std::floor(cv_d_max(1))) + 1,
      static_cast<int>(std::floor(cv_d_max(2))) + 1
    );

    if (static_cast<cell_index>(cv_max(0) - cv_min(0)) > mask ||
      static_cast<cell_index>(cv_max(1) - cv_min(1)) > mask ||
      static_cast<cell_index>(cv_max(2) - cv_min(2)) > mask)
      throw std::range_error("Bounds are too large or resolution is too small.");
  }

  geometry::vector3d cell_vector_d(const geometry::point3d& p) const {
    return { p.dot(b0_), p.dot(b1_), p.dot(b2_) };
  }

  cell_index cell_contains_point(const geometry::point3d& p) const {
    auto cv_d = cell_vector_d(p);
    auto cv0 = static_cast<int>(std::floor(cv_d(0)));
    auto cv1 = static_cast<int>(std::floor(cv_d(1)));
    auto cv2 = static_cast<int>(std::floor(cv_d(2)));

    cell_index offset2 = static_cast<cell_index>(cv2 - cv_min(2)) << shift2;
    cell_index offset21 = offset2 | (static_cast<cell_index>(cv1 - cv_min(1)) << shift1);
    return offset21 | static_cast<cell_index>(cv0 - cv_min(0));
  }

  cell_vector cell_vector_from_index(cell_index ci) const {
    auto cv0 = static_cast<int>(ci & mask) + cv_min(0);
    auto cv1 = static_cast<int>((ci >> shift1) & mask) + cv_min(1);
    auto cv2 = static_cast<int>((ci >> shift2) & mask) + cv_min(2);

    return { cv0, cv1, cv2 };
  }

  bool is_inside_bounds(const geometry::point3d& p) const {
    return
      p(0) >= ext_bbox_.min()(0) && p(0) <= ext_bbox_.max()(0) &&
      p(1) >= ext_bbox_.min()(1) && p(1) <= ext_bbox_.max()(1) &&
      p(2) >= ext_bbox_.min()(2) && p(2) <= ext_bbox_.max()(2);
  }

  geometry::bbox3d node_bounds() const {
    return ext_bbox_;
  }

  geometry::point3d point_from_cell_vector(const cell_vector& cv) const {
    return cv(0) * a0_ + cv(1) * a1_ + cv(2) * a2_;
  }

protected:
  cell_vector cv_min;
  cell_vector cv_max;
  const unsigned int shift1 = 21;
  const unsigned int shift2 = 42;
  const cell_index mask = (cell_index{ 1 } << shift1) - 1;

private:
  // Primitive vectors scaled by `lc`.
  geometry::vector3d a0_;
  geometry::vector3d a1_;
  geometry::vector3d a2_;

  // Reciprocal primitive vectors scaled by `rlc`.
  geometry::vector3d b0_;
  geometry::vector3d b1_;
  geometry::vector3d b2_;

  // Extended bbox
  geometry::bbox3d ext_bbox_;
};

}  // namespace isosurface
}  // namespace polatory
