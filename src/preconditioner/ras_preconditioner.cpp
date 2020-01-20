// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/preconditioner/ras_preconditioner.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/polynomial/orthonormal_basis.hpp>
#include <polatory/polynomial/unisolvent_point_set.hpp>
#include <polatory/preconditioner/domain_divider.hpp>

namespace polatory {
namespace preconditioner {

ras_preconditioner::ras_preconditioner(const model& model, const geometry::points3d& in_points)
  : model_without_poly_(model.without_poly())
  , points_(in_points)
  , n_points_(static_cast<index_t>(in_points.rows()))
  , n_poly_basis_(model.poly_basis_size())
  , finest_evaluator_(kReportResidual ? std::make_unique<interpolation::rbf_symmetric_evaluator<Order>>(model, points_) : nullptr)
{
  std::vector<index_t> poly_point_idcs;

  if (n_poly_basis_ > 0) {
    polynomial::unisolvent_point_set ups(points_, model.poly_dimension(), model.poly_degree());

    poly_point_idcs = ups.point_indices();
    lagrange_basis_ = std::make_unique<polynomial::lagrange_basis>(model.poly_dimension(), model.poly_degree(), common::take_rows(points_, poly_point_idcs));

    point_idcs_.push_back(poly_point_idcs);
    point_idcs_.back().reserve(n_points_);
    for (index_t i = 0; i < n_points_; i++) {
      if (!std::binary_search(poly_point_idcs.begin(), poly_point_idcs.end(), i)) {
        point_idcs_.back().push_back(i);
      }
    }
  } else {
    point_idcs_.emplace_back(n_points_);
    std::iota(point_idcs_.back().begin(), point_idcs_.back().end(), 0);
  }

  n_fine_levels_ = std::max(0, static_cast<int>(
    std::ceil(std::log(static_cast<double>(n_points_) / static_cast<double>(n_coarsest_points)) / log(1.0 / coarse_ratio))));
  if (n_fine_levels_ == 0) {
    coarse_ = std::make_unique<coarse_grid>(model, lagrange_basis_, point_idcs_.back(), points_);
    return;
  }

  auto bbox = geometry::bbox3d::from_points(points_);
  auto divider = std::make_unique<domain_divider>(points_, point_idcs_.back(), poly_point_idcs);

  fine_grids_.emplace_back();
  for (const auto& d : divider->domains()) {
    fine_grids_.back().emplace_back(model, lagrange_basis_, d.point_indices, d.inner_point);
  }
  auto n_points = static_cast<index_t>(points_.rows());
  auto n_fine_grids = static_cast<index_t>(fine_grids_.back().size());
  if (!kRecomputeAndClear) {
#pragma omp parallel for
    for (index_t i = 0; i < n_fine_grids; i++) {
      auto& fine = fine_grids_.back()[i];
      fine.setup(points_);
    }
  }
  std::cout << "Number of points in level 0: " << n_points << std::endl;
  std::cout << "Number of domains in level 0: " << n_fine_grids << std::endl;

  auto ratio = 0 == n_fine_levels_ - 1
               ? static_cast<double>(n_coarsest_points) / static_cast<double>(n_points)
               : coarse_ratio;
  upward_evaluator_.emplace_back(model_without_poly_, points_, bbox);
  point_idcs_.push_back(divider->choose_coarse_points(ratio));
  upward_evaluator_.back().set_field_points(common::take_rows(points_, point_idcs_.back()));

  for (auto level = 1; level < n_fine_levels_; level++) {
    divider = std::make_unique<domain_divider>(points_, point_idcs_.back(), poly_point_idcs);

    fine_grids_.emplace_back();
    for (const auto& d : divider->domains()) {
      fine_grids_.back().emplace_back(model, lagrange_basis_, d.point_indices, d.inner_point);
    }
    n_points = static_cast<index_t>(point_idcs_.back().size());
    n_fine_grids = static_cast<index_t>(fine_grids_.back().size());
    if (!kRecomputeAndClear) {
#pragma omp parallel for
      for (index_t i = 0; i < n_fine_grids; i++) {
        auto& fine = fine_grids_.back()[i];
        fine.setup(points_);
      }
    }
    std::cout << "Number of points in level " << level << ": " << n_points << std::endl;
    std::cout << "Number of domains in level " << level << ": " << n_fine_grids << std::endl;

    ratio = level == n_fine_levels_ - 1
            ? static_cast<double>(n_coarsest_points) / static_cast<double>(n_points)
            : coarse_ratio;
    upward_evaluator_.emplace_back(model_without_poly_, common::take_rows(points_, point_idcs_.back()), bbox);
    point_idcs_.push_back(divider->choose_coarse_points(ratio));
    upward_evaluator_.back().set_field_points(common::take_rows(points_, point_idcs_.back()));
  }

  n_points = static_cast<index_t>(point_idcs_.back().size());
  std::cout << "Number of points in coarse: " << n_points << std::endl;
  coarse_ = std::make_unique<coarse_grid>(model, lagrange_basis_, point_idcs_.back(), points_);

  for (auto level = 1; level < n_fine_levels_; level++) {
    downward_evaluator_.emplace_back(model, common::take_rows(points_, point_idcs_.back()), bbox);
    downward_evaluator_.back().set_field_points(common::take_rows(points_, point_idcs_[level]));
  }

  if (n_poly_basis_ > 0) {
    polynomial::orthonormal_basis poly(model.poly_dimension(), model.poly_degree(), points_);
    p_ = poly.evaluate(points_).transpose();
    ap_ = Eigen::MatrixXd(p_.rows(), p_.cols());

    auto finest_evaluator = interpolation::rbf_symmetric_evaluator<Order>(model_without_poly_, points_);
    auto n_cols = static_cast<index_t>(p_.cols());
    for (index_t i = 0; i < n_cols; i++) {
      finest_evaluator.set_weights(p_.col(i));
      ap_.col(i) = finest_evaluator.evaluate();
    }
  }
}

common::valuesd ras_preconditioner::operator()(const common::valuesd& v) const {
  POLATORY_ASSERT(static_cast<index_t>(v.rows()) == size());

  common::valuesd residuals = v.head(n_points_);
  common::valuesd weights_total = common::valuesd::Zero(size());
  if (n_fine_levels_ == 0) {
    coarse_->solve(residuals);
    coarse_->set_solution_to(weights_total);
    return weights_total;
  }

  if (kReportResidual) {
    std::cout << "Initial residual: " << residuals.norm() << std::endl;
  }

  for (auto level = 0; level < n_fine_levels_; level++) {
    {
      common::valuesd weights = common::valuesd::Zero(n_points_);

      // Solve on subdomains.
      auto n_fine_grids = static_cast<index_t>(fine_grids_[level].size());
#pragma omp parallel for schedule(guided)
      for (index_t i = 0; i < n_fine_grids; i++) {
        auto& fine = fine_grids_[level][i];
        if (kRecomputeAndClear) {
          fine.setup(points_);
        }
        fine.solve(residuals);
        fine.set_solution_to(weights);
        if (kRecomputeAndClear) {
          fine.clear();
        }
      }

      // Evaluate residuals at coarse points.
      if (level > 0) {
        const auto& finer_indices = point_idcs_[level];
        auto n_finer_points = static_cast<index_t>(finer_indices.size());
        common::valuesd finer_weights(n_finer_points);
        for (index_t i = 0; i < n_finer_points; i++) {
          finer_weights(i) = weights(finer_indices[i]);
        }
        upward_evaluator_[level].set_weights(finer_weights);
      } else {
        upward_evaluator_[level].set_weights(weights);
      }
      auto fit = upward_evaluator_[level].evaluate();

      const auto& indices = point_idcs_[level + 1];
      auto n_points = static_cast<index_t>(indices.size());
      for (index_t i = 0; i < n_points; i++) {
        residuals(indices[i]) -= fit(i);
      }

      if (n_poly_basis_ > 0) {
        // Orthogonalize weights against P.
        auto n_cols = static_cast<index_t>(p_.cols());
        for (index_t i = 0; i < n_cols; i++) {
          auto dot = p_.col(i).dot(weights);
          weights -= dot * p_.col(i);
          residuals += dot * ap_.col(i);
        }
      }

      weights_total.head(n_points_) += weights;

      if (kReportResidual) {
         // Test residual
         finest_evaluator_->set_weights(weights_total);
         common::valuesd test_residuals = v.head(n_points_) - finest_evaluator_->evaluate();
         std::cout << "Residual after level " << level << ": " << test_residuals.norm() << std::endl;
      }
    }

    {
      common::valuesd weights = common::valuesd::Zero(n_points_ + n_poly_basis_);

      // Solve on coarse.
      coarse_->solve(residuals);
      coarse_->set_solution_to(weights);

      if (level < n_fine_levels_ - 1) {
        const auto& coarse_indices = point_idcs_.back();
        auto n_coarse_points = static_cast<index_t>(coarse_indices.size());
        common::valuesd coarse_weights(n_coarse_points + n_poly_basis_);
        for (index_t i = 0; i < n_coarse_points; i++) {
          coarse_weights(i) = weights(coarse_indices[i]);
        }
        coarse_weights.tail(n_poly_basis_) = weights.tail(n_poly_basis_);
        downward_evaluator_[level].set_weights(coarse_weights);

        auto fit = downward_evaluator_[level].evaluate();

        const auto& indices = point_idcs_[level + 1];
        auto n_points = static_cast<index_t>(indices.size());
        for (index_t i = 0; i < n_points; i++) {
          residuals(indices[i]) -= fit(i);
        }
      }

      weights_total += weights;

      if (kReportResidual) {
         // Test residual
         finest_evaluator_->set_weights(weights_total);
         common::valuesd test_residuals = v.head(n_points_) - finest_evaluator_->evaluate();
         std::cout << "Residual after coarse correction: " << test_residuals.norm() << std::endl;
      }
    }
  }

  return weights_total;
}

index_t ras_preconditioner::size() const {
  return n_points_ + n_poly_basis_;
}

}  // namespace preconditioner
}  // namespace polatory
