// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace kriging {

class empirical_variogram {
public:
  empirical_variogram(const geometry::points3d& points, const common::valuesd& values,
                      double bin_width, size_t n_bins);

  const std::vector<double>& bin_distance() const;

  const std::vector<size_t>& bin_num_pairs() const;

  const std::vector<double>& bin_variance() const;

  size_t num_bins() const;

private:
  double bin_width_;
  size_t n_bins_;

  std::vector<double> distance_;
  std::vector<size_t> num_pairs_;
  std::vector<double> variance_;
};

}  // namespace kriging
}  // namespace polatory
