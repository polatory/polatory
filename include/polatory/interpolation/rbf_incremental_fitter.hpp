// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>
#include <vector>

#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace interpolation {

class rbf_incremental_fitter {
  static constexpr size_t min_n_points_for_incremental_fitting = 4096;
  static constexpr double initial_points_ratio = 0.01;
  static constexpr double incremental_points_ratio = 0.1;

public:
  rbf_incremental_fitter(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                         const geometry::points3d& points);

  std::pair<std::vector<size_t>, common::valuesd>
  fit(const common::valuesd& values, double absolute_tolerance) const;

private:
  std::pair<std::vector<size_t>, common::valuesd> initial_point_indices_and_weights() const;

  std::vector<size_t> point_indices_complement(const std::vector<size_t>& point_indices) const;

  const rbf::rbf rbf_;
  const int poly_dimension_;
  const int poly_degree_;
  const geometry::points3d& points_;

  const size_t n_points_;
  const size_t n_poly_basis_;

  const geometry::bbox3d bbox_;
};

}  // namespace interpolation
}  // namespace polatory
