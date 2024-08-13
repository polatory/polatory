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

using polatory::geometry::bbox3d;
using polatory::geometry::cuboid3d;
using polatory::geometry::matrix3d;
using polatory::geometry::point3d;
using polatory::geometry::transform_vector;
using polatory::geometry::vector3d;
using polatory::isosurface::bit_count;
using polatory::isosurface::bit_pop;
using polatory::isosurface::rmt::edge_bitset;
using polatory::isosurface::rmt::edge_index;
using polatory::isosurface::rmt::inv_sqrt2;
using polatory::isosurface::rmt::kLatticeBasis;
using polatory::isosurface::rmt::kNeighborLatticeCoordinatesDeltas;
using polatory::isosurface::rmt::kNeighborMasks;
using polatory::isosurface::rmt::kOppositeEdge;
using polatory::isosurface::rmt::lattice_coordinates;
using polatory::isosurface::rmt::primitive_lattice;
using polatory::point_cloud::random_points;

TEST(rmt, lattice) {
  point3d min(-1.0, -1.0, -1.0);
  point3d max(1.0, 1.0, 1.0);

  bbox3d bbox(min, max);
  double resolution = 0.01;

  primitive_lattice lat(bbox, resolution, matrix3d::Identity());

  auto points = random_points(cuboid3d(min, max), 100);

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
  matrix3d rot = (Eigen::AngleAxisd(-pi / 2.0, vector3d::UnitZ()) *
                  Eigen::AngleAxisd(-pi / 4.0, vector3d::UnitY()))
                     .toRotationMatrix();

  vector3d a0 = transform_vector<3>(rot, inv_sqrt2 * vector3d{-1.0, 1.0, 1.0});
  vector3d a1 = transform_vector<3>(rot, inv_sqrt2 * vector3d{1.0, -1.0, 1.0});
  vector3d a2 = transform_vector<3>(rot, inv_sqrt2 * vector3d{1.0, 1.0, -1.0});

  EXPECT_NEAR(0.0, (kLatticeBasis.row(0) - a0).norm(), 1e-15);
  EXPECT_NEAR(0.0, (kLatticeBasis.row(1) - a1).norm(), 1e-15);
  EXPECT_NEAR(0.0, (kLatticeBasis.row(2) - a2).norm(), 1e-15);
}

TEST(rmt, neighbor_cell_vectors) {
  vector3d a0 = inv_sqrt2 * vector3d{-1.0, 1.0, 1.0};
  vector3d a1 = inv_sqrt2 * vector3d{1.0, -1.0, 1.0};
  vector3d a2 = inv_sqrt2 * vector3d{1.0, 1.0, -1.0};

  std::array<vector3d, 14> neighbor_vectors{
      inv_sqrt2 * vector3d{-1.0, 1.0, 1.0},    // 0
      inv_sqrt2 * vector3d{0.0, 2.0, 0.0},     // 1
      inv_sqrt2 * vector3d{1.0, 1.0, -1.0},    // 2
      inv_sqrt2 * vector3d{0.0, 0.0, 2.0},     // 3
      inv_sqrt2 * vector3d{1.0, 1.0, 1.0},     // 4
      inv_sqrt2 * vector3d{2.0, 0.0, 0.0},     // 5
      inv_sqrt2 * vector3d{1.0, -1.0, 1.0},    // 6
      inv_sqrt2 * vector3d{1.0, -1.0, -1.0},   // 7
      inv_sqrt2 * vector3d{0.0, -2.0, 0.0},    // 8
      inv_sqrt2 * vector3d{-1.0, -1.0, 1.0},   // 9
      inv_sqrt2 * vector3d{0.0, 0.0, -2.0},    // A
      inv_sqrt2 * vector3d{-1.0, -1.0, -1.0},  // B
      inv_sqrt2 * vector3d{-2.0, 0.0, 0.0},    // C
      inv_sqrt2 * vector3d{-1.0, 1.0, -1.0},   // D
  };

  for (edge_index ei = 0; ei < 14; ei++) {
    const auto& lc = kNeighborLatticeCoordinatesDeltas.at(ei);
    vector3d v = lc(0) * a0 + lc(1) * a1 + lc(2) * a2;

    EXPECT_EQ(neighbor_vectors.at(ei), v);
  }
}

TEST(rmt, neighbor_masks) {
  vector3d a0 = inv_sqrt2 * vector3d{-1.0, 1.0, 1.0};
  vector3d a1 = inv_sqrt2 * vector3d{1.0, -1.0, 1.0};
  vector3d a2 = inv_sqrt2 * vector3d{1.0, 1.0, -1.0};

  for (std::size_t i = 0; i < kNeighborMasks.size(); i++) {
    auto mask = kNeighborMasks.at(i);
    auto count = bit_count(mask);
    EXPECT_TRUE(count == 4 || count == 6);

    const auto& lci = kNeighborLatticeCoordinatesDeltas.at(i);
    vector3d vi = lci(0) * a0 + lci(1) * a1 + lci(2) * a2;
    for (auto k = 0; k < count; k++) {
      auto j = bit_pop(&mask);
      const auto& lcj = kNeighborLatticeCoordinatesDeltas.at(j);
      vector3d vj = lcj(0) * a0 + lcj(1) * a1 + lcj(2) * a2;
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
  for (edge_index ei = 0; ei < 14; ei++) {
    EXPECT_EQ(-kNeighborLatticeCoordinatesDeltas.at(ei),
              kNeighborLatticeCoordinatesDeltas.at(kOppositeEdge.at(ei)));
  }
}
