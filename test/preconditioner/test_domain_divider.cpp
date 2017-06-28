// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "distribution_generator/spherical_distribution.hpp"
#include "preconditioner/domain_divider.hpp"

using namespace polatory::preconditioner;
using polatory::distribution_generator::spherical_distribution;

TEST(domain_divider, trivial)
{
   size_t n_points = 10000;
   Eigen::Vector3d center = Eigen::Vector3d::Zero();
   double radius = 1.0;

   auto points = spherical_distribution(n_points, center, radius);

   domain_divider divider(points);

   std::vector<size_t> inner_points;
   inner_points.reserve(n_points);
   for (const auto& d : divider.domains()) {
      for (size_t i = 0; i < d.size(); i++) {
         if (d.inner_point[i]) {
            inner_points.push_back(d.point_indices[i]);
         }
      }
   }
   ASSERT_EQ(n_points, inner_points.size());

   std::sort(inner_points.begin(), inner_points.end());
   ASSERT_EQ(inner_points.end(), std::unique(inner_points.begin(), inner_points.end()));

   auto coarse_ratio = 0.1;
   auto n_coarse_points = divider.choose_coarse_points(coarse_ratio).size();
   ASSERT_LE(0.95 * coarse_ratio * n_points, n_coarse_points);
   ASSERT_GE(1.05 * coarse_ratio * n_points, n_coarse_points);

   // TODO: Check at least one coarse point is chosen from each domain.
}
