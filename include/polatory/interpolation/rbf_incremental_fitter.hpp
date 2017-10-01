// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <utility>

#include <boost/range/irange.hpp>
#include <Eigen/Core>

#include "polatory/common/vector_view.hpp"
#include "polatory/common/zip_sort.hpp"
#include "polatory/fmm/tree_height.hpp"
#include "polatory/geometry/bbox3.hpp"
#include "polatory/polynomial/basis_base.hpp"
#include "polatory/rbf/rbf_base.hpp"
#include "rbf_solver.hpp"

namespace polatory {
namespace interpolation {

class rbf_incremental_fitter {
  const size_t min_n_points_for_incremental_fitting = 10000;
  const double initial_points_ratio = 0.01;
  const double incremental_points_ratio = 0.1;

  const rbf::rbf_base& rbf;
  const int poly_degree;
  const std::vector<Eigen::Vector3d>& points;

  const size_t n_points;
  const size_t n_polynomials;

  const geometry::bbox3d bbox;

  std::pair<std::vector<size_t>, Eigen::VectorXd> initial_point_indices_and_weights() const {
    std::vector<size_t> indices;
    Eigen::VectorXd weights;

    if (n_points < min_n_points_for_incremental_fitting) {
      indices = std::vector<size_t>(n_points);
      std::iota(indices.begin(), indices.end(), 0);

      weights = Eigen::VectorXd::Zero(n_points + n_polynomials);
    } else {
      size_t n_initial_points = initial_points_ratio * n_points;

      std::random_device rd;
      std::mt19937 gen(rd());

      indices = std::vector<size_t>(n_points);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), gen);

      indices.resize(n_initial_points);
      weights = Eigen::VectorXd::Zero(n_initial_points + n_polynomials);
    }

    return std::make_pair(std::move(indices), std::move(weights));
  }

  std::vector<size_t> point_indices_complement(const std::vector<size_t>& point_indices) const {
    auto indices = point_indices;
    std::sort(indices.begin(), indices.end());

    auto universe = boost::irange<size_t>(0, n_points);

    std::vector<size_t> indices_c;
    indices_c.reserve(n_points - indices.size());

    std::set_difference(universe.begin(), universe.end(),
                        indices.begin(), indices.end(), std::back_inserter(indices_c));

    return indices_c;
  }

  Eigen::VectorXd reduced_values(const Eigen::VectorXd& values, const std::vector<size_t>& indices) const {
    Eigen::VectorXd reduced(indices.size());

    for (size_t i = 0; i < indices.size(); i++) {
      reduced(i) = values(indices[i]);
    }

    return reduced;
  }

public:
  template <typename Container>
  rbf_incremental_fitter(const rbf::rbf_base& rbf, int poly_degree,
                         const Container& points)
    : rbf(rbf)
    , poly_degree(poly_degree)
    , points(points)
    , n_points(points.size())
    , n_polynomials(polynomial::basis_base::dimension(poly_degree))
    , bbox(geometry::bbox3d::from_points(points)) {
  }

  template <typename Derived>
  std::pair<std::vector<size_t>, Eigen::VectorXd>
  fit(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance) const {
    std::vector<size_t> indices;
    Eigen::VectorXd weights;

    std::tie(indices, weights) = initial_point_indices_and_weights();

    std::unique_ptr<rbf_solver> solver;
    std::unique_ptr<rbf_evaluator<>> res_eval;
    auto last_tree_height = 0;

    while (true) {
      auto reduced_points = common::make_view(points, indices);
      auto tree_height = fmm::tree_height(indices.size());

      if (tree_height != last_tree_height) {
        solver = std::make_unique<rbf_solver>(rbf, poly_degree, tree_height, bbox);
        res_eval = std::make_unique<rbf_evaluator<>>(rbf, poly_degree, tree_height, bbox);
        last_tree_height = tree_height;
      }

      solver->set_points(reduced_points);
      weights = solver->solve(reduced_values(values, indices), absolute_tolerance, weights);

      auto indices_c = point_indices_complement(indices);
      auto reduced_points_c = common::make_view(points, indices_c);

      if (indices_c.size() == 0) break;

      // Evaluate residuals at the rest of the points.

      res_eval->set_source_points(reduced_points);
      res_eval->set_weights(weights);

      std::vector<double> residuals_c;
      auto fit_c = res_eval->evaluate_points(reduced_points_c);
      auto values_c = reduced_values(values, indices_c);
      residuals_c.resize(indices_c.size());
      Eigen::VectorXd::Map(residuals_c.data(), indices_c.size()) = (values_c - fit_c).cwiseAbs();

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
      weights = Eigen::VectorXd::Zero(indices.size() + n_polynomials);
      weights.head(n_points_prev) = weights_prev.head(n_points_prev);
      weights.tail(n_polynomials) = weights_prev.tail(n_polynomials);
    }

    return std::make_pair(indices, weights);
  }
};

} // namespace interpolation
} // namespace polatory
