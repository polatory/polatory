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
public:
  empirical_variogram(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::VectorXd& values,
    double bin_width,
    size_t n_bins)
    : bin_width_(bin_width)
    , n_bins_(n_bins)
    , distance_(n_bins)
    , num_pairs_(n_bins)
    , variance_(n_bins) {
    auto n_points = points.size();
    assert(values.size() == n_points);

    std::vector<numeric::kahan_sum_accumulator<double>> var_sum(n_bins_);
    std::vector<numeric::kahan_sum_accumulator<double>> dist_sum(n_bins_);

    for (size_t i = 0; i < n_points - 1; i++) {
      for (size_t j = i + 1; j < n_points; j++) {
        auto dist = (points[i] - points[j]).norm();
        size_t bin = std::floor(dist / bin_width_);
        if (bin >= n_bins_) continue;

        var_sum[bin] += std::pow(values[i] - values[j], 2.0) / 2.0;
        dist_sum[bin] += dist;
        num_pairs_[bin]++;
      }
    }

    for (int i = 0; i < n_bins_; i++) {
      if (num_pairs_[i] == 0) continue;

      variance_[i] = var_sum[i].get() / num_pairs_[i];
      distance_[i] = dist_sum[i].get() / num_pairs_[i];
    }
  }

  const std::vector<double>& bin_distance() const {
    return distance_;
  }

  const std::vector<int>& bin_num_pairs() const {
    return num_pairs_;
  }

  const std::vector<double>& bin_variance() const {
    return variance_;
  }

  size_t num_bins() const {
    return n_bins_;
  }

private:
  double bin_width_;
  size_t n_bins_;

  std::vector<double> distance_;
  std::vector<int> num_pairs_;
  std::vector<double> variance_;
};

} // namespace kriging
} // namespace polatory
