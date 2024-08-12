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
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/neighbor.hpp>

namespace polatory::isosurface::rmt {

inline constexpr double inv_sqrt2 = 0.5 * std::numbers::sqrt2;

// A basis for the body-centered cubic lattice.
inline const std::array<geometry::vector3d, 3> kLatticeBasis{
    geometry::vector3d{inv_sqrt2, 1.0, 0.0}, geometry::vector3d{-inv_sqrt2, 0.0, 1.0},
    geometry::vector3d{inv_sqrt2, -1.0, 0.0}};

// The dual of the basis above, i.e., kDualLatticeBasis.at(i).dot(kLatticeBasis.at(j)) is
// 1.0 if i == j, otherwise 0.0.
inline const std::array<geometry::vector3d, 3> kDualLatticeBasis{
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
        a0_(geometry::transform_vector<3>(inv_aniso_, resolution_ * kLatticeBasis[0])),
        a1_(geometry::transform_vector<3>(inv_aniso_, resolution_ * kLatticeBasis[1])),
        a2_(geometry::transform_vector<3>(inv_aniso_, resolution_ * kLatticeBasis[2])),
        b0_(geometry::transform_vector<3>(trans_aniso_, kDualLatticeBasis[0] / resolution_)),
        b1_(geometry::transform_vector<3>(trans_aniso_, kDualLatticeBasis[1] / resolution_)),
        b2_(geometry::transform_vector<3>(trans_aniso_, kDualLatticeBasis[2] / resolution_)),
        lc_origin_(compute_lattice_coordinates_origin()),
        first_ext_bbox_(compute_extended_bbox(1)),
        second_ext_bbox_(compute_extended_bbox(2)) {}

  const geometry::bbox3d& bbox() const { return bbox_; }

  geometry::point3d position(const lattice_coordinates& lc) const {
    return (lc_origin_(0) + lc(0)) * a0_ + (lc_origin_(1) + lc(1)) * a1_ +
           (lc_origin_(2) + lc(2)) * a2_;
  }

  std::pair<int, int> first_lattice_coordinate_range(int lc1, int lc2) const {
    geometry::point3d point = (lc_origin_(1) + lc1) * a1_ + (lc_origin_(2) + lc2) * a2_;
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

    auto min = lattice_coordinates_unrounded(point + min_t * direction)(0);
    auto max = lattice_coordinates_unrounded(point + max_t * direction)(0);
    return {static_cast<int>(std::floor(min)), static_cast<int>(std::ceil(max))};
  }

  std::pair<int, int> second_lattice_coordinate_range(int lc2) const {
    std::vector<geometry::vector3d> vertices;

    geometry::vector3d normal = a0_.cross(a1_);
    auto d = -normal.dot((lc_origin_(2) + lc2) * a2_);
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
          vertices.emplace_back(p + t * pq);
        }
      }
    }

    auto min = std::numeric_limits<double>::infinity();
    auto max = -std::numeric_limits<double>::infinity();
    for (const auto& v : vertices) {
      auto lc = lattice_coordinates_unrounded(v);
      min = std::min(min, lc(1));
      max = std::max(max, lc(1));
    }
    return {static_cast<int>(std::floor(min)), static_cast<int>(std::ceil(max))};
  }

  std::pair<int, int> third_lattice_coordinate_range() const {
    auto vertices = second_extended_bbox().corners();
    auto min = std::numeric_limits<double>::infinity();
    auto max = -std::numeric_limits<double>::infinity();
    for (auto v : vertices.rowwise()) {
      auto lc = lattice_coordinates_unrounded(v);
      min = std::min(min, lc(2));
      max = std::max(max, lc(2));
    }
    return {static_cast<int>(std::floor(min)), static_cast<int>(std::ceil(max))};
  }

  lattice_coordinates lattice_coordinates_rounded(const geometry::point3d& p) const {
    return lattice_coordinates_unrounded(p).array().round().cast<int>();
  }

  // Returns a positively-oriented tetrahedron containing the given point.
  Eigen::Matrix<int, 4, 3, Eigen::RowMajor> tetrahedron(const geometry::point3d& p) const {
    auto lcd = lattice_coordinates_unrounded(p);
    geometry::vector3d lcd_int = lcd.array().floor();
    geometry::vector3d lcd_frac = lcd - lcd_int;
    lattice_coordinates lc = lcd_int.cast<int>();
    std::array<int, 3> rank = {0, 1, 2};
    std::sort(rank.begin(), rank.end(), [&](auto i, auto j) { return lcd_frac(i) < lcd_frac(j); });

    Eigen::Matrix<int, 4, 3, Eigen::RowMajor> tet;
    tet.row(0) = lc;
    auto make_class = [](int i, int j, int k) constexpr -> int { return 9 * i + 3 * j + k; };
    switch (make_class(rank.at(0), rank.at(1), rank.at(2))) {
      case make_class(2, 1, 0):
        tet.row(1) = neighbor(lc, edge::k0);
        tet.row(2) = neighbor(lc, edge::k3);
        break;
      case make_class(2, 0, 1):
        tet.row(1) = neighbor(lc, edge::k3);
        tet.row(2) = neighbor(lc, edge::k6);
        break;
      case make_class(0, 2, 1):
        tet.row(1) = neighbor(lc, edge::k6);
        tet.row(2) = neighbor(lc, edge::k5);
        break;
      case make_class(0, 1, 2):
        tet.row(1) = neighbor(lc, edge::k5);
        tet.row(2) = neighbor(lc, edge::k2);
        break;
      case make_class(1, 0, 2):
        tet.row(1) = neighbor(lc, edge::k2);
        tet.row(2) = neighbor(lc, edge::k1);
        break;
      case make_class(1, 2, 0):
        tet.row(1) = neighbor(lc, edge::k1);
        tet.row(2) = neighbor(lc, edge::k0);
        break;
      default:
        POLATORY_UNREACHABLE();
        break;
    }
    tet.row(3) = neighbor(lc, edge::k4);
    return tet;
  }

  // Returns the bounding box that contains all nodes to be clustered.
  const geometry::bbox3d& first_extended_bbox() const { return first_ext_bbox_; }

  double resolution() const { return resolution_; }

  // Returns the bounding box that contains all nodes eligible to be added to the lattice.
  const geometry::bbox3d& second_extended_bbox() const { return second_ext_bbox_; }

 private:
  geometry::vector3d lattice_coordinates_unrounded(const geometry::point3d& p) const {
    return {p.dot(b0_) - lc_origin_(0), p.dot(b1_) - lc_origin_(1), p.dot(b2_) - lc_origin_(2)};
  }

  geometry::vector3d compute_lattice_coordinates_origin() const {
    geometry::point3d center = bbox_.center();
    return {std::round(center.dot(b0_)), std::round(center.dot(b1_)), std::round(center.dot(b2_))};
  }

  geometry::bbox3d compute_extended_bbox(int extension) const {
    geometry::vectors3d neigh(14, 3);
    for (edge_index ei = 0; ei < 14; ei++) {
      auto lc = kNeighborLatticeCoordinatesDeltas.at(ei);
      neigh.row(ei) = lc(0) * a0_ + lc(1) * a1_ + lc(2) * a2_;
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
  const geometry::vector3d lc_origin_;
  const geometry::bbox3d first_ext_bbox_;
  const geometry::bbox3d second_ext_bbox_;
};

}  // namespace polatory::isosurface::rmt
