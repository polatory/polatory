// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>
#include <utility>

#include <boost/range/irange.hpp>
#include <Eigen/Core>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/quasi_random_sequence.hpp>
#include <polatory/common/types.hpp>
#include <polatory/common/zip_sort.hpp>
#include <polatory/fmm/tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace interpolation {

class rbf_incremental_fitter {
  const size_t min_n_points_for_incremental_fitting = 10000;
  const double initial_points_ratio = 0.01;
  const double incremental_points_ratio = 0.1;

public:
  rbf_incremental_fitter(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                         const geometry::points3d& points)
    : rbf_(rbf)
    , poly_dimension_(poly_dimension)
    , poly_degree_(poly_degree)
    , points_(points)
    , n_points_(points.rows())
    , n_poly_basis_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , bbox_(geometry::bbox3d::from_points(points)) {
  }

  template <class Derived>
  std::pair<std::vector<size_t>, common::valuesd>
  fit(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance) const {
    std::vector<size_t> indices;
    common::valuesd weights;

    std::tie(indices, weights) = initial_point_indices_and_weights();

    std::unique_ptr<rbf_solver> solver;
    std::unique_ptr<rbf_evaluator<>> res_eval;
    auto last_tree_height = 0;

    while (true) {
      auto reduced_points = common::take_rows(points_, indices);
      auto tree_height = fmm::tree_height(indices.size());

      if (tree_height != last_tree_height) {
        solver = std::make_unique<rbf_solver>(rbf_, poly_dimension_, poly_degree_, tree_height, bbox_);
        res_eval = std::make_unique<rbf_evaluator<>>(rbf_, poly_dimension_, poly_degree_, tree_height, bbox_);
        last_tree_height = tree_height;
      }

      solver->set_points(reduced_points);
      weights = solver->solve(common::take_rows(values, indices), absolute_tolerance, weights);

      auto indices_c = point_indices_complement(indices);
      auto reduced_points_c = common::take_rows(points_, indices_c);

      if (indices_c.size() == 0) break;

      // Evaluate residuals at the rest of the points.

      res_eval->set_source_points(reduced_points);
      res_eval->set_weights(weights);

      std::vector<double> residuals_c;
      auto fit_c = res_eval->evaluate_points(reduced_points_c);
      auto values_c = common::take_rows(values, indices_c);
      residuals_c.resize(indices_c.size());
      common::valuesd::Map(residuals_c.data(), indices_c.size()) = (values_c - fit_c).cwiseAbs();

      // Sort point indices by their residuals.

      common::zip_sort(
        indices_c.begin(), indices_c.end(),
        residuals_c.begin(), residuals_c.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
      );
      std::cout << residuals_c.front() << std::endl;
      std::cout << residuals_c.back() << std::endl;

      // Count points with residuals larger than absolute_tolerance.

      auto it = std::lower_bound(residuals_c.begin(), residuals_c.end(), absolute_tolerance);
      size_t n_points_need_fitting = std::distance(it, residuals_c.end());
      std::cout << n_points_need_fitting << " / " << indices_c.size() << std::endl;

      if (n_points_need_fitting == 0) break;

      // Append points with the largest residuals.

      auto n_points_prev = indices.size();
      auto n_points_to_add =
        std::min(n_points_need_fitting,
                 std::max(size_t(1000), static_cast<size_t>(n_points_need_fitting * incremental_points_ratio)));

      indices.insert(indices.end(), indices_c.end() - n_points_to_add, indices_c.end());

      auto weights_prev = weights;
      weights = common::valuesd::Zero(indices.size() + n_poly_basis_);
      weights.head(n_points_prev) = weights_prev.head(n_points_prev);
      weights.tail(n_poly_basis_) = weights_prev.tail(n_poly_basis_);
    }

    return std::make_pair(indices, weights);
  }

private:
  std::pair<std::vector<size_t>, common::valuesd> initial_point_indices_and_weights() const {
    std::vector<size_t> indices;
    common::valuesd weights;

    if (n_points_ < min_n_points_for_incremental_fitting) {
      indices = std::vector<size_t>(n_points_);
      std::iota(indices.begin(), indices.end(), 0);

      weights = common::valuesd::Zero(n_points_ + n_poly_basis_);
    } else {
      size_t n_initial_points = initial_points_ratio * n_points_;

      indices = common::quasi_random_sequence(n_points_);
      indices.resize(n_initial_points);

      weights = common::valuesd::Zero(n_initial_points + n_poly_basis_);
    }

    return std::make_pair(std::move(indices), std::move(weights));
  }

  std::vector<size_t> point_indices_complement(const std::vector<size_t>& point_indices) const {
    auto indices = point_indices;
    std::sort(indices.begin(), indices.end());

    auto universe = boost::irange<size_t>(0, n_points_);

    std::vector<size_t> indices_c(n_points_ - indices.size());
    std::set_difference(universe.begin(), universe.end(),
                        indices.begin(), indices.end(), indices_c.begin());

    return indices_c;
  }

  const rbf::rbf rbf_;
  const int poly_dimension_;
  const int poly_degree_;
  const geometry::points3d& points_;

  const size_t n_points_;
  const size_t n_poly_basis_;

  const geometry::bbox3d bbox_;
};

} // namespace interpolation
} // namespace polatory
