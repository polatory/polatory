#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <boost/container_hash/hash.hpp>
#include <cmath>
#include <functional>
#include <numbers>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/types.hpp>
#include <stdexcept>

namespace polatory::isosurface::rmt {

using cell_vector = Eigen::Vector3i;
using cell_vectors = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct cell_vector_hash {
  std::size_t operator()(const cell_vector& cv) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, std::hash<int>()(cv(0)));
    boost::hash_combine(seed, std::hash<int>()(cv(1)));
    boost::hash_combine(seed, std::hash<int>()(cv(2)));
    return seed;
  }
};

inline geometry::matrix3d rotation() {
  return geometry::to_matrix3d(
      Eigen::AngleAxisd(-std::numbers::pi / 2.0, geometry::vector3d::UnitZ()) *
      Eigen::AngleAxisd(-std::numbers::pi / 4.0, geometry::vector3d::UnitY()));
}

// Primitive vectors of the body-centered cubic lattice.
inline const std::array<geometry::vector3d, 3> kLatticeVectors{
    {geometry::transform_vector<3>(rotation(), geometry::vector3d{-1.0, 1.0, 1.0}) /
         std::numbers::sqrt2,
     geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, -1.0, 1.0}) /
         std::numbers::sqrt2,
     geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, 1.0, -1.0}) /
         std::numbers::sqrt2}};

// Reciprocal primitive vectors of the body-centered cubic lattice.
inline const std::array<geometry::vector3d, 3> kDualLatticeVectors{
    {geometry::transform_vector<3>(rotation(), geometry::vector3d{0.0, 1.0, 1.0}) /
         std::numbers::sqrt2,
     geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, 0.0, 1.0}) /
         std::numbers::sqrt2,
     geometry::transform_vector<3>(rotation(), geometry::vector3d{1.0, 1.0, 0.0}) /
         std::numbers::sqrt2}};

class primitive_lattice {
 public:
  primitive_lattice(const geometry::bbox3d& bbox, double resolution)
      : a0_(resolution * kLatticeVectors[0]),
        a1_(resolution * kLatticeVectors[1]),
        a2_(resolution * kLatticeVectors[2]),
        b0_(kDualLatticeVectors[0] / resolution),
        b1_(kDualLatticeVectors[1] / resolution),
        b2_(kDualLatticeVectors[2] / resolution),
        bbox_(bbox),
        ext_bbox_(compute_extended_bbox(bbox, resolution)),
        cv_offset_(compute_cv_offset()),
        resolution_(resolution) {
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

  const geometry::bbox3d& bbox() const { return bbox_; }

  geometry::point3d cell_node_point(const cell_vector& cv) const {
    return (cv_offset_(0) + static_cast<double>(cv(0))) * a0_ +
           (cv_offset_(1) + static_cast<double>(cv(1))) * a1_ +
           (cv_offset_(2) + static_cast<double>(cv(2))) * a2_;
  }

  cell_vector cell_vector_from_point(const geometry::point3d& p) const {
    return {static_cast<int>(std::floor(p.dot(b0_)) - cv_offset_(0)),
            static_cast<int>(std::floor(p.dot(b1_)) - cv_offset_(1)),
            static_cast<int>(std::floor(p.dot(b2_)) - cv_offset_(2))};
  }

  // All nodes in the extended bbox must be evaluated
  // to ensure that the isosurface does not have boundary in the bbox.
  const geometry::bbox3d& extended_bbox() const { return ext_bbox_; }

  double resolution() const { return resolution_; }

 protected:
  cell_vector cv_min;
  cell_vector cv_max;

 private:
  geometry::vector3d compute_cv_offset() const {
    geometry::point3d center = bbox_.center();
    return {std::floor(center.dot(b0_)), std::floor(center.dot(b1_)), std::floor(center.dot(b2_))};
  }

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
  const geometry::vector3d cv_offset_;
  const double resolution_;
};

}  // namespace polatory::isosurface::rmt
