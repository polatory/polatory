// Copyright (c) 2016, GSI and The Polatory Authors.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <polatory/common/types.hpp>
#include <polatory/geometry/cuboid3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/point_cloud/random_points.hpp>

using polatory::common::valuesd;
using polatory::geometry::cuboid3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::kriging::empirical_variogram;
using polatory::point_cloud::random_points;

TEST(empirical_variogram, serialize) {
  const size_t n_points = 100;

  points3d points = random_points(cuboid3d(), n_points);
  valuesd values = valuesd::Random(n_points);

  auto filename = (boost::filesystem::temp_directory_path() / boost::filesystem::unique_path())
    .string();

  empirical_variogram v(points, values, 0.1, 10);
  v.save(filename);

  empirical_variogram v2(filename);

  ASSERT_TRUE(std::equal(v.bin_distance().begin(), v.bin_distance().end(),
                         v2.bin_distance().begin(), v2.bin_distance().end()));

  ASSERT_TRUE(std::equal(v.bin_gamma().begin(), v.bin_gamma().end(),
                         v2.bin_gamma().begin(), v2.bin_gamma().end()));

  ASSERT_TRUE(std::equal(v.bin_num_pairs().begin(), v.bin_num_pairs().end(),
                         v2.bin_num_pairs().begin(), v2.bin_num_pairs().end()));
}

TEST(empirical_variogram, trivial) {
  const size_t n_points = 4;

  // Tetrahedron vertices separated from each other by a distance d.
  double d = 0.5;
  points3d points(n_points, 3);
  points <<
    d * point3d(std::sqrt(3.0) / 3.0, 0.0, 0.0),
    d * point3d(-std::sqrt(3.0) / 6.0, 1.0 / 2.0, 0.0),
    d * point3d(-std::sqrt(3.0) / 6.0, -1.0 / 2.0, 0.0),
    d * point3d(0.0, 0.0, std::sqrt(6.0) / 3.0);

  valuesd values = valuesd::Random(n_points);
  valuesd centered = values.array() - values.mean();
  double variance = centered.dot(centered) / static_cast<double>(n_points - 1);

  double bin_width = 0.2;
  int n_bins = 5;

  empirical_variogram variog(points, values, bin_width, n_bins);

  const auto bin_distance = variog.bin_distance();
  EXPECT_EQ(1u, bin_distance.size());
  EXPECT_DOUBLE_EQ(d, bin_distance[0]);

  const auto bin_gamma = variog.bin_gamma();
  EXPECT_EQ(1u, bin_gamma.size());
  EXPECT_DOUBLE_EQ(variance, bin_gamma[0]);

  const auto bin_num_pairs = variog.bin_num_pairs();
  EXPECT_EQ(1u, bin_num_pairs.size());
  EXPECT_EQ(6u, bin_num_pairs[0]);
}

TEST(empirical_variogram, zero_points) {
  const size_t n_points = 0;

  points3d points(n_points, 3);
  valuesd values(n_points);

  double bin_width = 0.2;
  int n_bins = 5;

  empirical_variogram variog(points, values, bin_width, n_bins);

  EXPECT_EQ(0u, variog.bin_distance().size());
  EXPECT_EQ(0u, variog.bin_gamma().size());
  EXPECT_EQ(0u, variog.bin_num_pairs().size());
}
