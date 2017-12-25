// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/kriging/empirical_variogram.hpp>

#include <cassert>
#include <cmath>

#include <polatory/numeric/sum_accumulator.hpp>

namespace polatory {
namespace kriging {

empirical_variogram::empirical_variogram(const geometry::points3d& points, const common::valuesd& values,
                                         double bin_width, size_t n_bins)
  : bin_width_(bin_width)
  , n_bins_(n_bins)
  , distance_(n_bins)
  , num_pairs_(n_bins)
  , variance_(n_bins) {
  auto n_points = points.rows();
  assert(values.size() == n_points);

  std::vector<numeric::kahan_sum_accumulator<double>> dist_sum(n_bins_);
  std::vector<numeric::kahan_sum_accumulator<double>> var_sum(n_bins_);

  for (size_t i = 0; i < n_points - 1; i++) {
    for (size_t j = i + 1; j < n_points; j++) {
      auto dist = (points.row(i) - points.row(j)).norm();
      // gstat's convention (to include more pairs in the first bin?):
      //   https://github.com/edzer/gstat/blob/a2644e4ff5af26e03feff89c033484486231f4bf/src/sem.c#L734
      auto frac = dist / bin_width_;
      size_t bin = dist > 0.0 && frac == std::floor(frac)
                   ? static_cast<size_t>(std::floor(frac) - 1)
                   : static_cast<size_t>(std::floor(frac));
      if (bin >= n_bins_) continue;

      dist_sum[bin] += dist;
      var_sum[bin] += std::pow(values[i] - values[j], 2.0) / 2.0;
      num_pairs_[bin]++;
    }
  }

  for (size_t i = 0; i < n_bins_; i++) {
    if (num_pairs_[i] == 0) continue;

    distance_[i] = dist_sum[i].get() / num_pairs_[i];
    variance_[i] = var_sum[i].get() / num_pairs_[i];
  }
}

const std::vector<double>& empirical_variogram::bin_distance() const {
  return distance_;
}

const std::vector<size_t>& empirical_variogram::bin_num_pairs() const {
  return num_pairs_;
}

const std::vector<double>& empirical_variogram::bin_variance() const {
  return variance_;
}

size_t empirical_variogram::num_bins() const {
  return n_bins_;
}

}  // namespace kriging
}  // namespace polatory
