// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/kriging/empirical_variogram.hpp>

using namespace polatory::kriging;

TEST(empirical_variogram, trivial) {
  int n_points = 4;

  // Tetrahedron vertices separated from each other by a distance d.
  double d = 0.5;
  std::vector<Eigen::Vector3d> points{
    d * Eigen::Vector3d(std::sqrt(3.0) / 3.0, 0.0, 0.0),
    d * Eigen::Vector3d(-std::sqrt(3.0) / 6.0, 1.0 / 2.0, 0.0),
    d * Eigen::Vector3d(-std::sqrt(3.0) / 6.0, -1.0 / 2.0, 0.0),
    d * Eigen::Vector3d(0.0, 0.0, std::sqrt(6.0) / 3.0)
  };

  Eigen::VectorXd values(n_points);
  values << 0.0, 1.0, 2.0, 3.0;

  int n_bins = 5;
  double bin_width = 0.2;
  int filled_bin = std::floor(d / bin_width);

  empirical_variogram variog(points, values, bin_width, n_bins);

  const auto bin_num_pairs = variog.bin_num_pairs();
  for (int bin = 0; bin < n_bins; bin++) {
    if (bin == filled_bin) {
      ASSERT_EQ(6u, bin_num_pairs[bin]);
    } else {
      ASSERT_EQ(0u, bin_num_pairs[bin]);
    }
  }

  const auto bin_distance = variog.bin_distance();
  ASSERT_DOUBLE_EQ(d, bin_distance[filled_bin]);

  const auto bin_variance = variog.bin_variance();
  double variance_expected =
    (std::pow(values(0) - values(1), 2.0) +
     std::pow(values(0) - values(2), 2.0) +
     std::pow(values(0) - values(3), 2.0) +
     std::pow(values(1) - values(2), 2.0) +
     std::pow(values(1) - values(3), 2.0) +
     std::pow(values(2) - values(3), 2.0)
    ) / (2.0 * 6.0);
  ASSERT_DOUBLE_EQ(variance_expected, bin_variance[filled_bin]);
}
