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
using polatory::isosurface::rmt::kDualLatticeVectors;
using polatory::isosurface::rmt::kLatticeVectors;
using polatory::isosurface::rmt::kNeighborCellVectors;
using polatory::isosurface::rmt::kNeighborMasks;
using polatory::isosurface::rmt::kOppositeEdge;
using polatory::isosurface::rmt::primitive_lattice;
using polatory::isosurface::rmt::rotation;
using polatory::point_cloud::random_points;

// Relative positions of neighbor nodes connected by each edge.
std::array<vector3d, 14> kNeighborVectors{
    transform_vector<3>(rotation(), vector3d{-1.0, 1.0, 1.0}) / std::numbers::sqrt2,    // 0
    transform_vector<3>(rotation(), vector3d{0.0, 2.0, 0.0}) / std::numbers::sqrt2,     // 1
    transform_vector<3>(rotation(), vector3d{1.0, 1.0, -1.0}) / std::numbers::sqrt2,    // 2
    transform_vector<3>(rotation(), vector3d{0.0, 0.0, 2.0}) / std::numbers::sqrt2,     // 3
    transform_vector<3>(rotation(), vector3d{1.0, 1.0, 1.0}) / std::numbers::sqrt2,     // 4
    transform_vector<3>(rotation(), vector3d{2.0, 0.0, 0.0}) / std::numbers::sqrt2,     // 5
    transform_vector<3>(rotation(), vector3d{1.0, -1.0, 1.0}) / std::numbers::sqrt2,    // 6
    transform_vector<3>(rotation(), vector3d{1.0, -1.0, -1.0}) / std::numbers::sqrt2,   // 7
    transform_vector<3>(rotation(), vector3d{0.0, -2.0, 0.0}) / std::numbers::sqrt2,    // 8
    transform_vector<3>(rotation(), vector3d{-1.0, -1.0, 1.0}) / std::numbers::sqrt2,   // 9
    transform_vector<3>(rotation(), vector3d{0.0, 0.0, -2.0}) / std::numbers::sqrt2,    // A
    transform_vector<3>(rotation(), vector3d{-1.0, -1.0, -1.0}) / std::numbers::sqrt2,  // B
    transform_vector<3>(rotation(), vector3d{-2.0, 0.0, 0.0}) / std::numbers::sqrt2,    // C
    transform_vector<3>(rotation(), vector3d{-1.0, +1.0, -1.0}) / std::numbers::sqrt2,  // D
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
