#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <numbers>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/bit.hpp>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>
#include <polatory/isosurface/rmt/node.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/point_cloud/random_points.hpp>

using polatory::Mat3;
using polatory::geometry::Bbox3;
using polatory::geometry::Cuboid3;
using polatory::geometry::Point3;
using polatory::geometry::transform_vector;
using polatory::geometry::Vector3;
using polatory::isosurface::bit_count;
using polatory::isosurface::bit_pop;
using polatory::isosurface::rmt::EdgeBitset;
using polatory::isosurface::rmt::EdgeIndex;
using polatory::isosurface::rmt::inv_sqrt2;
using polatory::isosurface::rmt::kLatticeBasis;
using polatory::isosurface::rmt::kNeighborLatticeCoordinatesDeltas;
using polatory::isosurface::rmt::kNeighborMasks;
using polatory::isosurface::rmt::kOppositeEdge;
using polatory::isosurface::rmt::LatticeCoordinates;
using polatory::isosurface::rmt::PrimitiveLattice;
using polatory::point_cloud::random_points;

TEST(rmt, lattice) {
  Point3 min(-1.0, -1.0, -1.0);
  Point3 max(1.0, 1.0, 1.0);

  Bbox3 bbox(min, max);
  double resolution = 0.01;

  PrimitiveLattice lat(bbox, resolution, Mat3::Identity());

  auto points = random_points(Cuboid3(min, max), 100);

  for (auto p : points.rowwise()) {
    auto lc = lat.lattice_coordinates_rounded(p);
    auto lp = lat.position(lc);

    EXPECT_LT((p - lp).norm(), std::sqrt(2.0) * resolution);
  }
}

TEST(rmt, construction_of_basis) {
  auto pi = std::numbers::pi;

  // Rotates the lattice so that the plane formed by the first and third primitive vectors
  // aligns with the xy-plane.
  Mat3 rot = (Eigen::AngleAxisd(-pi / 2.0, Vector3::UnitZ()) *
              Eigen::AngleAxisd(-pi / 4.0, Vector3::UnitY()))
                 .toRotationMatrix();

  Vector3 a0 = transform_vector<3>(rot, inv_sqrt2 * Vector3{-1.0, 1.0, 1.0});
  Vector3 a1 = transform_vector<3>(rot, inv_sqrt2 * Vector3{1.0, -1.0, 1.0});
  Vector3 a2 = transform_vector<3>(rot, inv_sqrt2 * Vector3{1.0, 1.0, -1.0});

  EXPECT_NEAR(0.0, (kLatticeBasis.row(0) - a0).norm(), 1e-15);
  EXPECT_NEAR(0.0, (kLatticeBasis.row(1) - a1).norm(), 1e-15);
  EXPECT_NEAR(0.0, (kLatticeBasis.row(2) - a2).norm(), 1e-15);
}

TEST(rmt, neighbor_cell_vectors) {
  Vector3 a0 = inv_sqrt2 * Vector3{-1.0, 1.0, 1.0};
  Vector3 a1 = inv_sqrt2 * Vector3{1.0, -1.0, 1.0};
  Vector3 a2 = inv_sqrt2 * Vector3{1.0, 1.0, -1.0};

  std::array<Vector3, 14> neighbor_vectors{
      inv_sqrt2 * Vector3{-1.0, 1.0, 1.0},    // 0
      inv_sqrt2 * Vector3{0.0, 2.0, 0.0},     // 1
      inv_sqrt2 * Vector3{1.0, 1.0, -1.0},    // 2
      inv_sqrt2 * Vector3{0.0, 0.0, 2.0},     // 3
      inv_sqrt2 * Vector3{1.0, 1.0, 1.0},     // 4
      inv_sqrt2 * Vector3{2.0, 0.0, 0.0},     // 5
      inv_sqrt2 * Vector3{1.0, -1.0, 1.0},    // 6
      inv_sqrt2 * Vector3{1.0, -1.0, -1.0},   // 7
      inv_sqrt2 * Vector3{0.0, -2.0, 0.0},    // 8
      inv_sqrt2 * Vector3{-1.0, -1.0, 1.0},   // 9
      inv_sqrt2 * Vector3{0.0, 0.0, -2.0},    // A
      inv_sqrt2 * Vector3{-1.0, -1.0, -1.0},  // B
      inv_sqrt2 * Vector3{-2.0, 0.0, 0.0},    // C
      inv_sqrt2 * Vector3{-1.0, 1.0, -1.0},   // D
  };

  for (EdgeIndex ei = 0; ei < 14; ei++) {
    const auto& lc = kNeighborLatticeCoordinatesDeltas.at(ei);
    Vector3 v = lc(0) * a0 + lc(1) * a1 + lc(2) * a2;

    EXPECT_EQ(neighbor_vectors.at(ei), v);
  }
}

TEST(rmt, neighbor_masks) {
  Vector3 a0 = inv_sqrt2 * Vector3{-1.0, 1.0, 1.0};
  Vector3 a1 = inv_sqrt2 * Vector3{1.0, -1.0, 1.0};
  Vector3 a2 = inv_sqrt2 * Vector3{1.0, 1.0, -1.0};

  for (std::size_t i = 0; i < kNeighborMasks.size(); i++) {
    auto mask = kNeighborMasks.at(i);
    auto count = bit_count(mask);
    EXPECT_TRUE(count == 4 || count == 6);

    const auto& lci = kNeighborLatticeCoordinatesDeltas.at(i);
    Vector3 vi = lci(0) * a0 + lci(1) * a1 + lci(2) * a2;
    for (auto k = 0; k < count; k++) {
      auto j = bit_pop(&mask);
      const auto& lcj = kNeighborLatticeCoordinatesDeltas.at(j);
      Vector3 vj = lcj(0) * a0 + lcj(1) * a1 + lcj(2) * a2;
      auto vij2 = (vj - vi).squaredNorm();
      if (vij2 > 1.75) {
        EXPECT_DOUBLE_EQ(2.0, vij2);
      } else {
        EXPECT_DOUBLE_EQ(1.5, vij2);
      }
    }
  }
}

TEST(rmt, opposite_edge) {
  for (EdgeIndex ei = 0; ei < 14; ei++) {
    EXPECT_EQ(-kNeighborLatticeCoordinatesDeltas.at(ei),
              kNeighborLatticeCoordinatesDeltas.at(kOppositeEdge.at(ei)));
  }
}
