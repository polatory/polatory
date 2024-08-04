#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <array>
#include <boost/container_hash/hash.hpp>
#include <cmath>
#include <numbers>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/rmt/neighbor.hpp>
#include <polatory/isosurface/rmt/types.hpp>

namespace polatory::isosurface::rmt {

inline constexpr double inv_sqrt2 = 0.5 * std::numbers::sqrt2;

// Primitive vectors of the body-centered cubic lattice.
inline const std::array<geometry::vector3d, 3> kLatticeVectors{
    geometry::vector3d{inv_sqrt2, 1.0, 0.0}, geometry::vector3d{-inv_sqrt2, 0.0, 1.0},
    geometry::vector3d{inv_sqrt2, -1.0, 0.0}};

// Reciprocal primitive vectors of the body-centered cubic lattice.
inline const std::array<geometry::vector3d, 3> kDualLatticeVectors{
    geometry::vector3d{inv_sqrt2, 0.5, 0.5}, geometry::vector3d{0.0, 0.0, 1.0},
    geometry::vector3d{inv_sqrt2, -0.5, 0.5}};

class primitive_lattice {
 public:
  primitive_lattice(const geometry::bbox3d& bbox, double resolution,
                    const geometry::matrix3d& aniso)
      : bbox_(bbox),
        resolution_(resolution),
        inv_aniso_(aniso.inverse()),
        trans_aniso_(aniso.transpose()),
        a0_(geometry::transform_vector<3>(inv_aniso_, resolution_ * kLatticeVectors[0])),
        a1_(geometry::transform_vector<3>(inv_aniso_, resolution_ * kLatticeVectors[1])),
        a2_(geometry::transform_vector<3>(inv_aniso_, resolution_ * kLatticeVectors[2])),
        b0_(geometry::transform_vector<3>(trans_aniso_, kDualLatticeVectors[0] / resolution_)),
        b1_(geometry::transform_vector<3>(trans_aniso_, kDualLatticeVectors[1] / resolution_)),
        b2_(geometry::transform_vector<3>(trans_aniso_, kDualLatticeVectors[2] / resolution_)),
        cv_offset_(compute_cv_offset()),
        first_ext_bbox_(compute_extended_bbox(1)),
        second_ext_bbox_(compute_extended_bbox(2)) {}

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

  cell_vector closest_cell_vector(const geometry::point3d& p) const {
    return {static_cast<int>(std::round(p.dot(b0_)) - cv_offset_(0)),
            static_cast<int>(std::round(p.dot(b1_)) - cv_offset_(1)),
            static_cast<int>(std::round(p.dot(b2_)) - cv_offset_(2))};
  }

  // Returns the bounding box that contains all nodes to be clustered.
  const geometry::bbox3d& first_extended_bbox() const { return first_ext_bbox_; }

  double resolution() const { return resolution_; }

  // Returns the bounding box that contains all nodes eligible to be added to the lattice.
  const geometry::bbox3d& second_extended_bbox() const { return second_ext_bbox_; }

 private:
  geometry::vector3d compute_cv_offset() const {
    geometry::point3d center = bbox_.center();
    return {std::floor(center.dot(b0_)), std::floor(center.dot(b1_)), std::floor(center.dot(b2_))};
  }

  geometry::bbox3d compute_extended_bbox(int extension) const {
    geometry::vectors3d neigh(14, 3);
    for (edge_index ei = 0; ei < 14; ei++) {
      auto cv = kNeighborCellVectors.at(ei);
      neigh.row(ei) = cv(0) * a0_ + cv(1) * a1_ + cv(2) * a2_;
    }

    geometry::vector3d min_ext = extension * 1.01 * neigh.colwise().minCoeff();
    geometry::vector3d max_ext = extension * 1.01 * neigh.colwise().maxCoeff();

    return {bbox_.min() + min_ext, bbox_.max() + max_ext};
  }

  const geometry::bbox3d bbox_;
  const double resolution_;
  const geometry::matrix3d inv_aniso_;
  const geometry::matrix3d trans_aniso_;
  const geometry::vector3d a0_;
  const geometry::vector3d a1_;
  const geometry::vector3d a2_;
  const geometry::vector3d b0_;
  const geometry::vector3d b1_;
  const geometry::vector3d b2_;
  const geometry::vector3d cv_offset_;
  const geometry::bbox3d first_ext_bbox_;
  const geometry::bbox3d second_ext_bbox_;
};

}  // namespace polatory::isosurface::rmt
