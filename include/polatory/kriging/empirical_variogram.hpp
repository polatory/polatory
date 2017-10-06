// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "polatory/numeric/sum_accumulator.hpp"

namespace polatory {
namespace kriging {

class empirical_variogram {
  const size_t n_points;
  const std::vector<Eigen::Vector3d>& points;
  const Eigen::VectorXd& values;

  double bin_range;
  int n_bins;

  std::vector<double> variances;
  std::vector<int> num_pairs;

public:
  empirical_variogram(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::VectorXd& values,
    double bin_range,
    double n_bins)
    : n_points(points.size())
    , points(points)
    , values(values)
    , bin_range(bin_range)
    , n_bins(n_bins)
    , variances(n_bins)
    , num_pairs(n_bins) {
    assert(values.size() == n_points);

    std::vector<numeric::kahan_sum_accumulator<double>> sums(n_bins);

    for (size_t i = 0; i < n_points - 1; i++) {
      for (size_t j = i + 1; j < n_points; j++) {
        double distance = (points[i] - points[j]).norm();
        double diff_sq = std::pow(values[i] - values[j], 2.0);
        int bin = std::floor(distance / bin_range);
        if (bin >= n_bins) continue;

        sums[bin] += diff_sq;
        num_pairs[bin]++;
      }
    }

    for (int i = 0; i < n_bins; i++) {
      if (num_pairs[i] == 0) continue;
      variances[i] = sums[i].get() / (2.0 * num_pairs[i]);
    }
  }

  std::vector<double> bin_centers() const {
    std::vector<double> centers(n_bins);

    for (int i = 0; i < n_bins; i++) {
      centers[i] = (i + 0.5) * bin_range;
    }

    return centers;
  }

  const std::vector<int>& bin_num_pairs() const {
    return num_pairs;
  }

  const std::vector<double>& bin_variances() const {
    return variances;
  }

  int num_bins() const {
    return n_bins;
  }
};

} // namespace kriging
} // namespace polatory
