// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/point_cloud/random_points.hpp>
#include <polatory/preconditioner/domain_divider.hpp>

using namespace polatory::preconditioner;
using polatory::geometry::sphere3d;
using polatory::point_cloud::random_points;

TEST(domain_divider, trivial) {
  size_t n_points = 10000;
  size_t n_poly_points = 10;

  auto points = random_points(sphere3d(), n_points);
  std::vector<size_t> point_idcs(n_points);
  std::iota(point_idcs.begin(), point_idcs.end(), 0);

  std::vector<size_t> poly_point_idcs(point_idcs.begin(), point_idcs.begin() + n_poly_points);

  domain_divider divider(points, point_idcs, poly_point_idcs);

  std::vector<size_t> inner_points;
  inner_points.reserve(n_points);
  for (const auto& d : divider.domains()) {
    for (size_t i = 0; i < d.size(); i++) {
      if (d.inner_point[i]) {
        inner_points.push_back(d.point_indices[i]);
      }
    }

    for (size_t i = 0; i < n_poly_points; i++) {
      ASSERT_EQ(poly_point_idcs[i], d.point_indices[i]);
    }

    auto domain_points = d.point_indices;
    std::sort(domain_points.begin(), domain_points.end());
    ASSERT_EQ(domain_points.end(), std::unique(domain_points.begin(), domain_points.end()));
  }
  ASSERT_EQ(n_points, inner_points.size());

  std::sort(inner_points.begin(), inner_points.end());
  ASSERT_EQ(inner_points.end(), std::unique(inner_points.begin(), inner_points.end()));

  auto coarse_ratio = 0.1;
  auto coarse_point_idcs = divider.choose_coarse_points(coarse_ratio);
  ASSERT_LE(0.95 * coarse_ratio * n_points, coarse_point_idcs.size());
  ASSERT_GE(1.05 * coarse_ratio * n_points, coarse_point_idcs.size());

  for (size_t i = 0; i < n_poly_points; i++) {
    ASSERT_EQ(poly_point_idcs[i], coarse_point_idcs[i]);
  }

  // TODO(mizuno): Check at least one coarse point is chosen from each domain.
}
