#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

namespace polatory {
namespace interpolation {

class rbf_incremental_fitter {
  static constexpr index_t min_n_points_for_incremental_fitting = 4096;
  static constexpr double initial_points_ratio = 0.01;
  static constexpr double point_adoption_ratio = 0.1;
  static constexpr index_t max_n_points_to_add = 1024;

 public:
  rbf_incremental_fitter(const model& model, const geometry::points3d& points);

  std::pair<std::vector<index_t>, common::valuesd> fit(const common::valuesd& values,
                                                       double absolute_tolerance) const;

 private:
  std::vector<index_t> initial_indices() const;

  std::vector<index_t> complement_indices(const std::vector<index_t>& indices) const;

  const model& model_;
  const geometry::points3d& points_;

  const index_t n_points_;
  const index_t n_poly_basis_;

  const geometry::bbox3d bbox_;
};

}  // namespace interpolation
}  // namespace polatory
