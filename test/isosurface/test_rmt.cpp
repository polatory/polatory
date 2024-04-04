#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/bit.hpp>
#include <polatory/isosurface/rmt/lattice.hpp>
#include <polatory/isosurface/rmt/node.hpp>
#include <polatory/isosurface/rmt/node_list.hpp>
#include <polatory/isosurface/rmt/primitive_lattice.hpp>
#include <polatory/point_cloud/random_points.hpp>

using polatory::geometry::bbox3d;
using polatory::geometry::cuboid3d;
using polatory::geometry::point3d;
using polatory::geometry::transform_vector;
using polatory::geometry::vector3d;
using polatory::isosurface::bit_count;
using polatory::isosurface::bit_pop;
using polatory::isosurface::rmt::edge_bitset;
using polatory::isosurface::rmt::edge_index;
using polatory::isosurface::rmt::inv_sqrt2;
using polatory::isosurface::rmt::kDualLatticeVectors;
using polatory::isosurface::rmt::kLatticeVectors;
using polatory::isosurface::rmt::kNeighborCellVectors;
using polatory::isosurface::rmt::kNeighborMasks;
using polatory::isosurface::rmt::kOppositeEdge;
using polatory::isosurface::rmt::primitive_lattice;
using polatory::isosurface::rmt::rotate;
using polatory::point_cloud::random_points;

// Relative positions of neighbor nodes connected by each edge.
std::array<vector3d, 14> kNeighborVectors{
    rotate(inv_sqrt2 * vector3d{-1.0, 1.0, 1.0}),   // 0
    rotate(inv_sqrt2* vector3d{0.0, 2.0, 0.0}),     // 1
    rotate(inv_sqrt2* vector3d{1.0, 1.0, -1.0}),    // 2
    rotate(inv_sqrt2* vector3d{0.0, 0.0, 2.0}),     // 3
    rotate(inv_sqrt2* vector3d{1.0, 1.0, 1.0}),     // 4
    rotate(inv_sqrt2* vector3d{2.0, 0.0, 0.0}),     // 5
    rotate(inv_sqrt2* vector3d{1.0, -1.0, 1.0}),    // 6
    rotate(inv_sqrt2* vector3d{1.0, -1.0, -1.0}),   // 7
    rotate(inv_sqrt2* vector3d{0.0, -2.0, 0.0}),    // 8
    rotate(inv_sqrt2* vector3d{-1.0, -1.0, 1.0}),   // 9
    rotate(inv_sqrt2* vector3d{0.0, 0.0, -2.0}),    // A
    rotate(inv_sqrt2* vector3d{-1.0, -1.0, -1.0}),  // B
    rotate(inv_sqrt2* vector3d{-2.0, 0.0, 0.0}),    // C
    rotate(inv_sqrt2* vector3d{-1.0, +1.0, -1.0}),  // D
};

TEST(rmt, lattice) {
  point3d min(-1.0, -1.0, -1.0);
  point3d max(1.0, 1.0, 1.0);

  bbox3d bbox(min, max);
  double resolution = 0.01;

  primitive_lattice lat(bbox, resolution);

  auto points = random_points(cuboid3d(min, max), 100);

  for (auto p : points.rowwise()) {
    auto cv = lat.cell_vector_from_point(p);
    auto cp = lat.cell_node_point(cv);

    EXPECT_LT((p - cp).norm(), std::sqrt(2.0) * resolution);
  }
}

TEST(rmt, lattice_vectors) {
  for (auto i = 0; i < 3; i++) {
    for (auto j = 0; j < 3; j++) {
      auto dot = kLatticeVectors.at(i).dot(kDualLatticeVectors.at(j));
      if (i == j) {
        EXPECT_NEAR(1.0, dot, 1e-15);
      } else {
        EXPECT_NEAR(0.0, dot, 1e-15);
      }
    }
  }
}

TEST(rmt, neighbors) {
  for (std::size_t i = 0; i < kNeighborMasks.size(); i++) {
    auto mask = kNeighborMasks.at(i);
    auto count = bit_count(mask);
    EXPECT_TRUE(count == 4 || count == 6);

    auto vi = kNeighborVectors.at(i);
    for (auto k = 0; k < count; k++) {
      auto j = bit_pop(&mask);
      auto vj = kNeighborVectors.at(j);
      auto vijsq = (vj - vi).squaredNorm();
      if (vijsq > 1.75) {
        EXPECT_DOUBLE_EQ(2.0, vijsq);
      } else {
        EXPECT_DOUBLE_EQ(1.5, vijsq);
      }
    }
  }

  for (edge_index ei = 0; ei < 14; ei++) {
    vector3d computed = kLatticeVectors[0] * kNeighborCellVectors.at(ei)[0] +
                        kLatticeVectors[1] * kNeighborCellVectors.at(ei)[1] +
                        kLatticeVectors[2] * kNeighborCellVectors.at(ei)[2];

    for (auto i = 0; i < 3; i++) {
      EXPECT_NEAR(kNeighborVectors.at(ei)(i), computed(i), 1e-15);
    }
  }
}

TEST(rmt, opposite_edge) {
  for (edge_index ei = 0; ei < 14; ei++) {
    EXPECT_EQ(-kNeighborVectors.at(ei), kNeighborVectors.at(kOppositeEdge.at(ei)));
  }
}
