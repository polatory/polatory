// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/interpolation/rbf_incremental_fitter.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>

#include <boost/range/irange.hpp>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/quasi_random_sequence.hpp>
#include <polatory/common/zip_sort.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/interpolation/rbf_solver.hpp>

namespace polatory {
namespace interpolation {

rbf_incremental_fitter::rbf_incremental_fitter(const rbf::rbf& rbf, const geometry::points3d& points)
  : rbf_(rbf)
  , points_(points)
  , n_points_(points.rows())
  , n_poly_basis_(rbf.poly_basis_size())
  , bbox_(geometry::bbox3d::from_points(points)) {
}

std::pair<std::vector<size_t>, common::valuesd>
rbf_incremental_fitter::fit(const common::valuesd& values, double absolute_tolerance) const {
  auto centers = initial_indices();
  common::valuesd center_weights = common::valuesd::Zero(centers.size() + n_poly_basis_);

  std::unique_ptr<rbf_solver> solver;
  std::unique_ptr<rbf_evaluator<>> res_eval;
  auto last_tree_height = 0;

  while (true) {
    std::cout << "Number of RBF centers: " << centers.size() << " / " << n_points_ << std::endl;

    auto center_points = common::take_rows(points_, centers);
    auto tree_height = fmm::fmm_tree_height(centers.size());

    if (tree_height != last_tree_height) {
      solver = std::make_unique<rbf_solver>(rbf_, tree_height, bbox_);
      res_eval = std::make_unique<rbf_evaluator<>>(rbf_, tree_height, bbox_);
      last_tree_height = tree_height;
    }

    solver->set_points(center_points);
    center_weights = solver->solve(common::take_rows(values, centers), absolute_tolerance, center_weights);

    if (centers.size() == n_points_)
      break;

    // Evaluate residuals at remaining points.

    auto c_centers = complement_indices(centers);
    auto c_center_points = common::take_rows(points_, c_centers);

    res_eval->set_source_points(center_points);
    res_eval->set_weights(center_weights);

    auto c_values_fit = res_eval->evaluate_points(c_center_points);
    auto c_values = common::take_rows(values, c_centers);
    std::vector<double> c_residuals(c_centers.size());
    common::valuesd::Map(c_residuals.data(), c_centers.size()) = (c_values_fit - c_values).cwiseAbs();

    // Sort remaining points by their residuals.

    common::zip_sort(
      c_centers.begin(), c_centers.end(),
      c_residuals.begin(), c_residuals.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    // Count points with residuals larger than absolute_tolerance.

    auto it = std::lower_bound(c_residuals.begin(), c_residuals.end(), absolute_tolerance);
    size_t n_points_need_fitting = std::distance(it, c_residuals.end());
    std::cout << "Number of points to fit: " << n_points_need_fitting << std::endl;

    if (n_points_need_fitting == 0)
      break;

    // Append points with the largest residuals.

    auto n_last_centers = centers.size();
    auto n_centers_to_add =
      std::min(n_points_need_fitting,
               std::max(size_t(max_n_points_to_add), static_cast<size_t>(n_points_need_fitting * point_adoption_ratio)));

    centers.insert(centers.end(), c_centers.end() - n_centers_to_add, c_centers.end());

    auto last_center_weights = center_weights;
    center_weights = common::valuesd::Zero(centers.size() + n_poly_basis_);
    center_weights.head(n_last_centers) = last_center_weights.head(n_last_centers);
    center_weights.tail(n_poly_basis_) = last_center_weights.tail(n_poly_basis_);
  }

  return std::make_pair(std::move(centers), std::move(center_weights));
}

std::vector<size_t> rbf_incremental_fitter::initial_indices() const {
  std::vector<size_t> idcs;

  if (n_points_ < min_n_points_for_incremental_fitting) {
    idcs.resize(n_points_);
    std::iota(idcs.begin(), idcs.end(), 0);
  } else {
    size_t n_initial_points = initial_points_ratio * n_points_;

    idcs = common::quasi_random_sequence(n_points_);
    idcs.resize(n_initial_points);
  }

  return idcs;
}

std::vector<size_t> rbf_incremental_fitter::complement_indices(const std::vector<size_t>& indices) const {
  std::vector<size_t> c_idcs(n_points_ - indices.size());

  auto universe = boost::irange<size_t>(0, n_points_);
  auto idcs = indices;
  std::sort(idcs.begin(), idcs.end());
  std::set_difference(universe.begin(), universe.end(),
                      idcs.begin(), idcs.end(), c_idcs.begin());

  return c_idcs;
}

}  // namespace interpolation
}  // namespace polatory
