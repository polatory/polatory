// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>
#include <vector>

#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>

namespace polatory {
namespace interpolation {

class rbf_incremental_fitter {
  static constexpr size_t min_n_points_for_incremental_fitting = 4096;
  static constexpr double initial_points_ratio = 0.01;
  static constexpr double point_adoption_ratio = 0.1;
  static constexpr size_t max_n_points_to_add = 1024;

public:
  rbf_incremental_fitter(const model& model, const geometry::points3d& points);

  std::pair<std::vector<size_t>, common::valuesd>
  fit(const common::valuesd& values, double absolute_tolerance) const;

private:
  std::vector<size_t> initial_indices() const;

  std::vector<size_t> complement_indices(const std::vector<size_t>& indices) const;

  const model model_;
  const geometry::points3d& points_;

  const size_t n_points_;
  const size_t n_poly_basis_;

  const geometry::bbox3d bbox_;
};

}  // namespace interpolation
}  // namespace polatory
