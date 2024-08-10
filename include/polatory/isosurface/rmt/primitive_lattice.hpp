#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/rmt/edge.hpp>
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

  std::pair<int, int> first_cell_vector_range(int m1, int m2) const {
    geometry::point3d point = m1 * a1_ + m2 * a2_;
    geometry::point3d direction = a0_;
    const auto& bbox = second_extended_bbox();

    auto min_t = -std::numeric_limits<double>::infinity();
    auto max_t = std::numeric_limits<double>::infinity();

    for (auto i = 0; i < 3; i++) {
      if (direction(i) == 0.0) {
        continue;
      }

      auto t0 = (bbox.min()(i) - point(i)) / direction(i);
      auto t1 = (bbox.max()(i) - point(i)) / direction(i);
      if (t0 > t1) {
        std::swap(t0, t1);
      }

      min_t = std::max(min_t, t0);
      max_t = std::min(max_t, t1);
    }

    auto min = cell_vector_from_point_unrounded(point + min_t * direction)(0);
    auto max = cell_vector_from_point_unrounded(point + max_t * direction)(0);
    return {static_cast<int>(std::floor(min)), static_cast<int>(std::ceil(max))};
  }

  std::pair<int, int> second_cell_vector_range(int m2) const {
    std::vector<geometry::vector3d> vertices;

    geometry::vector3d normal = a0_.cross(a1_);
    auto d = -normal.dot(m2 * a2_);
    auto bbox_vertices = second_extended_bbox().corners();
    for (auto i = 0; i < 7; i++) {
      for (auto j = i + 1; j < 8; j++) {
        if (std::popcount(static_cast<unsigned>(i) ^ static_cast<unsigned>(j)) != 1) {
          // Not an edge.
          continue;
        }

        geometry::point3d p = bbox_vertices.row(i);
        geometry::point3d q = bbox_vertices.row(j);
        geometry::vector3d pq = q - p;
        auto t = -(normal.dot(p) + d) / normal.dot(pq);
        if (t >= 0 && t <= 1) {
          vertices.push_back(p + t * pq);
        }
      }
    }

    auto min = std::numeric_limits<double>::infinity();
    auto max = -std::numeric_limits<double>::infinity();
    for (auto v : vertices) {
      auto cv = cell_vector_from_point_unrounded(v);
      min = std::min(min, cv(1));
      max = std::max(max, cv(1));
    }
    return {static_cast<int>(std::floor(min)), static_cast<int>(std::ceil(max))};
  }

  std::pair<int, int> third_cell_vector_range() const {
    auto vertices = second_extended_bbox().corners();
    auto min = std::numeric_limits<double>::infinity();
    auto max = -std::numeric_limits<double>::infinity();
    for (auto v : vertices.rowwise()) {
      auto cv = cell_vector_from_point_unrounded(v);
      min = std::min(min, cv(2));
      max = std::max(max, cv(2));
    }
    return {static_cast<int>(std::floor(min)), static_cast<int>(std::ceil(max))};
  }

  cell_vector closest_cell_vector(const geometry::point3d& p) const {
    return cell_vector_from_point_unrounded(p).array().round().cast<int>();
  }

  // Returns a positively-oriented tetrahedron containing the given point.
  cell_vectors tetrahedron(const geometry::point3d& p) const {
    cell_vectors cvs(4, 3);
    auto cvd = cell_vector_from_point_unrounded(p);
    geometry::vector3d cvd_int = cvd.array().floor();
    geometry::vector3d cvd_frac = cvd - cvd_int;
    cell_vector cv = cvd_int.cast<int>();
    std::array<int, 3> rank = {0, 1, 2};
    std::sort(rank.begin(), rank.end(), [&](auto i, auto j) { return cvd_frac(i) < cvd_frac(j); });

    cvs.row(0) = cv;
    auto make_class = [](int i, int j, int k) constexpr -> int { return 9 * i + 3 * j + k; };
    switch (make_class(rank.at(0), rank.at(1), rank.at(2))) {
      case make_class(2, 1, 0):
        cvs.row(1) = neighbor(cv, edge::k0);
        cvs.row(2) = neighbor(cv, edge::k3);
        break;
      case make_class(2, 0, 1):
        cvs.row(1) = neighbor(cv, edge::k3);
        cvs.row(2) = neighbor(cv, edge::k6);
        break;
      case make_class(0, 2, 1):
        cvs.row(1) = neighbor(cv, edge::k6);
        cvs.row(2) = neighbor(cv, edge::k5);
        break;
      case make_class(0, 1, 2):
        cvs.row(1) = neighbor(cv, edge::k5);
        cvs.row(2) = neighbor(cv, edge::k2);
        break;
      case make_class(1, 0, 2):
        cvs.row(1) = neighbor(cv, edge::k2);
        cvs.row(2) = neighbor(cv, edge::k1);
        break;
      case make_class(1, 2, 0):
        cvs.row(1) = neighbor(cv, edge::k1);
        cvs.row(2) = neighbor(cv, edge::k0);
        break;
      default:
        POLATORY_UNREACHABLE();
        break;
    }
    cvs.row(3) = neighbor(cv, edge::k4);
    return cvs;
  }

  // Returns the bounding box that contains all nodes to be clustered.
  const geometry::bbox3d& first_extended_bbox() const { return first_ext_bbox_; }

  double resolution() const { return resolution_; }

  // Returns the bounding box that contains all nodes eligible to be added to the lattice.
  const geometry::bbox3d& second_extended_bbox() const { return second_ext_bbox_; }

 private:
  geometry::vector3d cell_vector_from_point_unrounded(const geometry::point3d& p) const {
    return {p.dot(b0_) - cv_offset_(0), p.dot(b1_) - cv_offset_(1), p.dot(b2_) - cv_offset_(2)};
  }

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
