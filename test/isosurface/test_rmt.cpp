// Copyright (c) 2016, GSI and The Polatory Authors.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "geometry/bbox3.hpp"
#include "isosurface/rmt_lattice.hpp"
#include "random_points/box_points.hpp"

using namespace polatory::isosurface;
using polatory::geometry::bbox3d;
using polatory::random_points::box_points;

// Relative positions of neighbor nodes connected by each edge.
std::array<Eigen::Vector3d, 14> NeighborVectors
  {
    rotation() * Eigen::Vector3d(+1., +1., +1.),
    rotation() * Eigen::Vector3d(+2., +0., +0.),
    rotation() * Eigen::Vector3d(+1., -1., -1.),
    rotation() * Eigen::Vector3d(+0., +2., +0.),
    rotation() * Eigen::Vector3d(+1., +1., -1.),
    rotation() * Eigen::Vector3d(+0., +0., -2.),
    rotation() * Eigen::Vector3d(-1., +1., -1.),
    rotation() * Eigen::Vector3d(-1., -1., -1.),
    rotation() * Eigen::Vector3d(-2., +0., +0.),
    rotation() * Eigen::Vector3d(-1., +1., +1.),
    rotation() * Eigen::Vector3d(+0., -2., +0.),
    rotation() * Eigen::Vector3d(-1., -1., +1.),
    rotation() * Eigen::Vector3d(+0., +0., +2.),
    rotation() * Eigen::Vector3d(+1., -1., +1.)
  };

TEST(rmt, face_edges) {
  for (edge_bitset edge_set : FaceEdges) {
    auto e0 = bit::pop(edge_set);
    auto e1 = bit::pop(edge_set);
    auto e2 = bit::pop(edge_set);

    auto& v0 = NeighborVectors[e0];
    auto& v1 = NeighborVectors[e1];
    auto& v2 = NeighborVectors[e2];

    auto area = (v1 - v0).cross(v2 - v0).norm();
    ASSERT_DOUBLE_EQ(2.0 * std::sqrt(2.0), area);
  }
}

TEST(rmt, lattice) {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double radius = 1.0;

  bbox3d bbox(center.array() - radius, center.array() + radius);
  double resolution = 0.01;

  rmt_primitive_lattice lat(bbox, resolution);

  auto points = box_points(100, center, radius);

  for (const auto& p : points) {
    auto ci = lat.cell_contains_point(p);
    auto cv = lat.cell_vector_from_index(ci);
    auto cp = lat.point_from_cell_vector(cv);

    ASSERT_LT((p - cp).norm(), std::sqrt(2.0) * resolution);
  }
}

TEST(rmt, neighbor_edge_pairs) {
  for (edge_index ei = 0; ei < 14; ei++) {
    auto& va = NeighborVectors[ei];

    for (auto& pair : NeighborEdgePairs[ei]) {
      auto& vb = NeighborVectors[pair.first];
      auto& vc = NeighborVectors[pair.second];

      ASSERT_TRUE(va.dot(vb) > 0.0);
      ASSERT_TRUE(va.dot(vc) > 0.0);
      ASSERT_DOUBLE_EQ(0.0, vb.cross(va).cross(va.cross(vc)).norm());
    }
  }
}

TEST(rmt, neighbors) {
  for (size_t i = 0; i < NeighborMasks.size(); i++) {
    auto mask = NeighborMasks[i];
    auto count = bit::count(mask);
    ASSERT_TRUE(count == 4 || count == 6);

    auto vi = NeighborVectors[i];
    for (int k = 0; k < count; k++) {
      auto j = bit::pop(mask);
      auto vj = NeighborVectors[j];
      double vijsq = (vj - vi).squaredNorm();
      if (vijsq > 3.5) {
        ASSERT_DOUBLE_EQ(4.0, vijsq);
      } else {
        ASSERT_DOUBLE_EQ(3.0, vijsq);
      }
    }
  }

  for (edge_index ei = 0; ei < 14; ei++) {
    Eigen::Vector3d computed =
      PrimitiveVectors[0] * NeighborCellVectors[ei][0]
      + PrimitiveVectors[1] * NeighborCellVectors[ei][1]
      + PrimitiveVectors[2] * NeighborCellVectors[ei][2];

    for (int i = 0; i < 3; i++) {
      ASSERT_DOUBLE_EQ(NeighborVectors[ei](i), computed(i));
    }
  }
}

TEST(rmt, opposite_edge) {
  for (edge_index ei = 0; ei < 14; ei++) {
    ASSERT_EQ(-NeighborVectors[ei], NeighborVectors[OppositeEdge[ei]]);
  }
}

TEST(rmt, primitive_vectors) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      auto dot = PrimitiveVectors[i].dot(ReciprocalPrimitiveVectors[j]);
      if (i == j) {
        ASSERT_NEAR(1.0, dot, 1e-15);
      } else {
        ASSERT_NEAR(0.0, dot, 1e-15);
      }
    }
  }
}

TEST(rmt, rotation) {
  Eigen::Matrix3d rtr = rotation().transpose() * rotation();
  ASSERT_LT((Eigen::Matrix3d::Identity() - rtr).lpNorm<Eigen::Infinity>(), 1e-12);
}
