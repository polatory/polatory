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
inline const Mat3 kLatticeBasis(
    (Mat3() << inv_sqrt2, 1.0, 0.0, -inv_sqrt2, 0.0, 1.0, inv_sqrt2, -1.0, 0.0).finished());

class PrimitiveLattice {
 public:
  PrimitiveLattice(const geometry::Bbox3& bbox, double resolution, const Mat3& aniso)
      : bbox_(bbox),
        resolution_(resolution),
        basis_(geometry::transform_vectors<3>(aniso.inverse(), resolution_ * kLatticeBasis)),
        basis_inv_(basis_.inverse()),
        lc_origin_(compute_lattice_coordinates_origin()),
        first_ext_bbox_(compute_extended_bbox(1)),
        second_ext_bbox_(compute_extended_bbox(2)) {}

  const geometry::Bbox3& bbox() const { return bbox_; }

  geometry::Point3 position(const LatticeCoordinates& lc) const {
    return (lc_origin_ + lc.cast<double>()) * basis_;
  }

  std::pair<int, int> first_lattice_coordinate_range(int lc1, int lc2) const {
    geometry::Point3 point =
        (lc_origin_(1) + lc1) * basis_.row(1) + (lc_origin_(2) + lc2) * basis_.row(2);
    geometry::Vector3 direction = basis_.row(0);
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
    std::vector<geometry::Vector3> vertices;

    geometry::Vector3 normal = basis_.row(0).cross(basis_.row(1));
    auto d = -normal.dot((lc_origin_(2) + lc2) * basis_.row(2));
    auto bbox_vertices = second_extended_bbox().corners();
    for (auto i = 0; i < 7; i++) {
      for (auto j = i + 1; j < 8; j++) {
        if (std::popcount(static_cast<unsigned>(i) ^ static_cast<unsigned>(j)) != 1) {
          // Not an edge.
          continue;
        }

        geometry::Point3 p = bbox_vertices.row(i);
        geometry::Point3 q = bbox_vertices.row(j);
        geometry::Vector3 pq = q - p;
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

  LatticeCoordinates lattice_coordinates_rounded(const geometry::Point3& p) const {
    return lattice_coordinates_unrounded(p).array().round().cast<int>();
  }

  // Returns a positively-oriented tetrahedron containing the given point.
  Eigen::Matrix<int, 4, 3, Eigen::RowMajor> tetrahedron(const geometry::Point3& p) const {
    auto lcd = lattice_coordinates_unrounded(p);
    geometry::Vector3 lcd_int = lcd.array().floor();
    geometry::Vector3 lcd_frac = lcd - lcd_int;
    LatticeCoordinates lc = lcd_int.cast<int>();
    std::array<int, 3> rank = {0, 1, 2};
    std::sort(rank.begin(), rank.end(), [&](auto i, auto j) { return lcd_frac(i) < lcd_frac(j); });

    Eigen::Matrix<int, 4, 3, Eigen::RowMajor> tet;
    tet.row(0) = lc;
    auto make_class = [](int i, int j, int k) constexpr -> int { return 9 * i + 3 * j + k; };
    switch (make_class(rank.at(0), rank.at(1), rank.at(2))) {
      case make_class(2, 1, 0):
        tet.row(1) = neighbor(lc, Edge::k0);
        tet.row(2) = neighbor(lc, Edge::k3);
        break;
      case make_class(2, 0, 1):
        tet.row(1) = neighbor(lc, Edge::k3);
        tet.row(2) = neighbor(lc, Edge::k6);
        break;
      case make_class(0, 2, 1):
        tet.row(1) = neighbor(lc, Edge::k6);
        tet.row(2) = neighbor(lc, Edge::k5);
        break;
      case make_class(0, 1, 2):
        tet.row(1) = neighbor(lc, Edge::k5);
        tet.row(2) = neighbor(lc, Edge::k2);
        break;
      case make_class(1, 0, 2):
        tet.row(1) = neighbor(lc, Edge::k2);
        tet.row(2) = neighbor(lc, Edge::k1);
        break;
      case make_class(1, 2, 0):
        tet.row(1) = neighbor(lc, Edge::k1);
        tet.row(2) = neighbor(lc, Edge::k0);
        break;
      default:
        POLATORY_UNREACHABLE();
        break;
    }
    tet.row(3) = neighbor(lc, Edge::k4);
    return tet;
  }

  // Returns the bounding box that contains all nodes to be clustered.
  const geometry::Bbox3& first_extended_bbox() const { return first_ext_bbox_; }

  double resolution() const { return resolution_; }

  // Returns the bounding box that contains all nodes eligible to be added to the lattice.
  const geometry::Bbox3& second_extended_bbox() const { return second_ext_bbox_; }

 private:
  geometry::Vector3 lattice_coordinates_unrounded(const geometry::Point3& p) const {
    return p * basis_inv_ - lc_origin_;
  }

  geometry::Vector3 compute_lattice_coordinates_origin() const {
    geometry::Point3 center = bbox_.center();
    return (center * basis_inv_).array().round();
  }

  geometry::Bbox3 compute_extended_bbox(int extension) const {
    geometry::Vectors3 neigh(14, 3);
    for (EdgeIndex ei = 0; ei < 14; ei++) {
      const auto& lc = kNeighborLatticeCoordinatesDeltas.at(ei);
      neigh.row(ei) = lc.cast<double>() * basis_;
    }

    geometry::Vector3 min_ext = extension * 1.01 * neigh.colwise().minCoeff();
    geometry::Vector3 max_ext = extension * 1.01 * neigh.colwise().maxCoeff();

    return {bbox_.min() + min_ext, bbox_.max() + max_ext};
  }

  const geometry::Bbox3 bbox_;
  const double resolution_;
  const Mat3 basis_;
  const Mat3 basis_inv_;
  const geometry::Vector3 lc_origin_;
  const geometry::Bbox3 first_ext_bbox_;
  const geometry::Bbox3 second_ext_bbox_;
};

}  // namespace polatory::isosurface::rmt
