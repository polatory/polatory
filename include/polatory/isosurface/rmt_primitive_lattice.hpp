#pragma once

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <numbers>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>
#include <stdexcept>

namespace polatory::isosurface {

namespace detail {

class lattice_vectors : public std::array<geometry::vector3d, 3> {
  using base = std::array<geometry::vector3d, 3>;

 public:
  lattice_vectors();
};

class dual_lattice_vectors : public std::array<geometry::vector3d, 3> {
  using base = std::array<geometry::vector3d, 3>;

 public:
  dual_lattice_vectors();
};

}  // namespace detail

inline geometry::linear_transformation3d rotation() {
  return geometry::to_linear_transformation3d(
      Eigen::AngleAxisd(-std::numbers::pi / 2.0, geometry::vector3d::UnitZ()) *
      Eigen::AngleAxisd(-std::numbers::pi / 4.0, geometry::vector3d::UnitY()));
}

// Primitive vectors of the body-centered cubic lattice.
extern const detail::lattice_vectors LatticeVectors;

// Reciprocal primitive vectors of the body-centered cubic lattice.
extern const detail::dual_lattice_vectors DualLatticeVectors;

class rmt_primitive_lattice {
 public:
  rmt_primitive_lattice(const geometry::bbox3d& bbox, double resolution)
      : a0_(resolution * LatticeVectors[0]),
        a1_(resolution * LatticeVectors[1]),
        a2_(resolution * LatticeVectors[2]),
        b0_(DualLatticeVectors[0] / resolution),
        b1_(DualLatticeVectors[1] / resolution),
        b2_(DualLatticeVectors[2] / resolution),
        bbox_(bbox),
        ext_bbox_(compute_extended_bbox(bbox, resolution)) {
    geometry::points3d ext_bbox_vertices(8, 3);
    ext_bbox_vertices << ext_bbox_.min()(0), ext_bbox_.min()(1), ext_bbox_.min()(2),
        ext_bbox_.max()(0), ext_bbox_.min()(1), ext_bbox_.min()(2), ext_bbox_.min()(0),
        ext_bbox_.max()(1), ext_bbox_.min()(2), ext_bbox_.min()(0), ext_bbox_.min()(1),
        ext_bbox_.max()(2), ext_bbox_.min()(0), ext_bbox_.max()(1), ext_bbox_.max()(2),
        ext_bbox_.max()(0), ext_bbox_.min()(1), ext_bbox_.max()(2), ext_bbox_.max()(0),
        ext_bbox_.max()(1), ext_bbox_.min()(2), ext_bbox_.max()(0), ext_bbox_.max()(1),
        ext_bbox_.max()(2);

    cell_vectors cvs(8, 3);
    for (auto i = 0; i < 8; i++) {
      cvs.row(i) = cell_vector_from_point(ext_bbox_vertices.row(i));
    }

    // Bounds of cell vectors for enumerating all nodes in the extended bbox.
    cv_min = cvs.colwise().minCoeff().array() + 1;
    cv_max = cvs.colwise().maxCoeff();
  }

  geometry::bbox3d bbox() const { return bbox_; }

  geometry::point3d cell_node_point(const cell_vector& cv) const {
    return cv(0) * a0_ + cv(1) * a1_ + cv(2) * a2_;
  }

  cell_vector cell_vector_from_point(const geometry::point3d& p) const {
    return {static_cast<int>(std::floor(p.dot(b0_))), static_cast<int>(std::floor(p.dot(b1_))),
            static_cast<int>(std::floor(p.dot(b2_)))};
  }

  // All nodes in the extended bbox must be evaluated
  // to ensure that the isosurface does not have boundary in the bbox.
  geometry::bbox3d extended_bbox() const { return ext_bbox_; }

 protected:
  cell_vector cv_min;
  cell_vector cv_max;

 private:
  static geometry::bbox3d compute_extended_bbox(const geometry::bbox3d& bbox, double resolution) {
    geometry::vector3d cell_bbox_size =
        resolution * geometry::vector3d(3.0 / std::numbers::sqrt2, 2.0, 1.0);
    geometry::vector3d ext = 1.01 * cell_bbox_size;
    return {bbox.min() - ext, bbox.max() + ext};
  }

  const geometry::vector3d a0_;
  const geometry::vector3d a1_;
  const geometry::vector3d a2_;
  const geometry::vector3d b0_;
  const geometry::vector3d b1_;
  const geometry::vector3d b2_;
  const geometry::bbox3d bbox_;
  const geometry::bbox3d ext_bbox_;
};

}  // namespace polatory::isosurface
