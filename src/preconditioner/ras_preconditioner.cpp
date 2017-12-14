// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/preconditioner/ras_preconditioner.hpp>

#include <numeric>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/polynomial/orthonormal_basis.hpp>
#include <polatory/polynomial/unisolvent_point_set.hpp>
#include <polatory/preconditioner/domain_divider.hpp>

namespace polatory {
namespace preconditioner {

ras_preconditioner::ras_preconditioner(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                                       const geometry::points3d& in_points)
  : points_(in_points)
  , n_points_(in_points.rows())
  , n_poly_basis_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
#if POLATORY_REPORT_RESIDUAL
  , finest_evaluator_(rbf, poly_dimension, poly_degree, points_)
#endif
{
  point_idcs_.push_back(std::vector<size_t>(n_points_));
  std::iota(point_idcs_.back().begin(), point_idcs_.back().end(), 0);

  std::vector<size_t> poly_point_idcs;
  if (n_poly_basis_ > 0) {
    polynomial::unisolvent_point_set ups(points_, point_idcs_.back(), poly_dimension, poly_degree);
    point_idcs_.back() = ups.point_indices();
    poly_point_idcs = std::vector<size_t>(point_idcs_.back().begin(), point_idcs_.back().begin() + n_poly_basis_);
    lagrange_basis_ = std::make_shared<polynomial::lagrange_basis>(poly_dimension, poly_degree, common::take_rows(points_, poly_point_idcs));
  }

  n_fine_levels_ = std::max(0, int(
    std::ceil(std::log(double(n_points_) / double(n_coarsest_points)) / log(1.0 / coarse_ratio))));
  if (n_fine_levels_ == 0) {
    coarse_ = std::make_unique<coarse_grid>(rbf, lagrange_basis_, point_idcs_.back(), points_);
    return;
  }

  auto bbox = geometry::bbox3d::from_points(points_);
  auto divider = std::make_unique<domain_divider>(points_, point_idcs_.back(), poly_point_idcs);

  fine_grids_.push_back(std::vector<fine_grid>());
  for (const auto& d : divider->domains()) {
    fine_grids_.back().push_back(fine_grid(rbf, lagrange_basis_, d.point_indices, d.inner_point));
  }
#if !POLATORY_RECOMPUTE_AND_CLEAR
#pragma omp parallel for
  for (size_t i = 0; i < fine_grids_.back().size(); i++) {
     auto& fine = fine_grids_.back()[i];
     fine.setup(points_);
  }
#endif
  std::cout << "Number of points in level 0: " << points_.rows() << std::endl;
  std::cout << "Number of domains in level 0: " << fine_grids_.back().size() << std::endl;

  auto ratio = 0 == n_fine_levels_ - 1
               ? double(n_coarsest_points) / double(points_.rows())
               : coarse_ratio;
  upward_evaluator_.push_back(interpolation::rbf_evaluator<Order>(rbf, -1, -1, points_, bbox));
  point_idcs_.push_back(divider->choose_coarse_points(ratio));
  upward_evaluator_.back().set_field_points(common::take_rows(points_, point_idcs_.back()));

  for (int level = 1; level < n_fine_levels_; level++) {
    divider = std::make_unique<domain_divider>(points_, point_idcs_.back(), poly_point_idcs);

    fine_grids_.push_back(std::vector<fine_grid>());
    for (const auto& d : divider->domains()) {
      fine_grids_.back().push_back(fine_grid(rbf, lagrange_basis_, d.point_indices, d.inner_point));
    }
#if !POLATORY_RECOMPUTE_AND_CLEAR
#pragma omp parallel for
    for (size_t i = 0; i < fine_grids_.back().size(); i++) {
       auto& fine = fine_grids_.back()[i];
       fine.setup(points_);
    }
#endif
    std::cout << "Number of points in level " << level << ": " << point_idcs_.back().size() << std::endl;
    std::cout << "Number of domains in level " << level << ": " << fine_grids_.back().size() << std::endl;

    ratio = level == n_fine_levels_ - 1
            ? double(n_coarsest_points) / double(point_idcs_.back().size())
            : coarse_ratio;
    upward_evaluator_.push_back(
      interpolation::rbf_evaluator<Order>(rbf, -1, -1, common::take_rows(points_, point_idcs_.back()), bbox));
    point_idcs_.push_back(divider->choose_coarse_points(ratio));
    upward_evaluator_.back().set_field_points(common::take_rows(points_, point_idcs_.back()));
  }

  std::cout << "Number of points in coarse: " << point_idcs_.back().size() << std::endl;
  coarse_ = std::make_unique<coarse_grid>(rbf, lagrange_basis_, point_idcs_.back(), points_);

  for (int level = 1; level < n_fine_levels_; level++) {
    downward_evaluator_.push_back(
      interpolation::rbf_evaluator<Order>(rbf, poly_dimension, poly_degree, common::take_rows(points_, point_idcs_.back()), bbox));
    downward_evaluator_.back().set_field_points(common::take_rows(points_, point_idcs_[level]));
  }

  if (n_poly_basis_ > 0) {
    polynomial::orthonormal_basis poly(poly_dimension, poly_degree, points_);
    p_ = poly.evaluate_points(points_).transpose();
    ap_ = Eigen::MatrixXd(p_.rows(), p_.cols());

    auto finest_evaluator = interpolation::rbf_symmetric_evaluator<Order>(rbf, -1, -1, points_);
    for (size_t i = 0; i < p_.cols(); i++) {
      finest_evaluator.set_weights(p_.col(i));
      ap_.col(i) = finest_evaluator.evaluate();
    }
  }
}

common::valuesd ras_preconditioner::operator()(const common::valuesd& v) const {
  assert(v.rows() == size());

  common::valuesd residuals = v.head(n_points_);
  common::valuesd weights_total = common::valuesd::Zero(size());
  if (n_fine_levels_ == 0) {
    coarse_->solve(residuals);
    coarse_->set_solution_to(weights_total);
    return weights_total;
  }

#if POLATORY_REPORT_RESIDUAL
  std::cout << "Initial residual: " << residuals.norm() << std::endl;
#endif

  for (int level = 0; level < n_fine_levels_; level++) {
    {
      common::valuesd weights = common::valuesd::Zero(n_points_);

      // Solve on subdomains.
#pragma omp parallel for schedule(guided)
      for (size_t i = 0; i < fine_grids_[level].size(); i++) {
        auto& fine = fine_grids_[level][i];
#if POLATORY_RECOMPUTE_AND_CLEAR
        fine.setup(points_);
#endif
        fine.solve(residuals);
        fine.set_solution_to(weights);
#if POLATORY_RECOMPUTE_AND_CLEAR
        fine.clear();
#endif
      }

      // Evaluate residuals at coarse points.
      if (level > 0) {
        const auto& finer_indices = point_idcs_[level];
        common::valuesd finer_weights(finer_indices.size());
        for (size_t i = 0; i < finer_indices.size(); i++) {
          finer_weights(i) = weights(finer_indices[i]);
        }
        upward_evaluator_[level].set_weights(finer_weights);
      } else {
        upward_evaluator_[level].set_weights(weights);
      }
      auto fit = upward_evaluator_[level].evaluate();

      const auto& indices = point_idcs_[level + 1];
      for (size_t i = 0; i < indices.size(); i++) {
        residuals(indices[i]) -= fit(i);
      }

      if (n_poly_basis_ > 0) {
        // Orthogonalize weights against P.
        for (size_t i = 0; i < p_.cols(); i++) {
          auto dot = p_.col(i).dot(weights);
          weights -= dot * p_.col(i);
          residuals += dot * ap_.col(i);
        }
      }

      weights_total.head(n_points_) += weights;

#if POLATORY_REPORT_RESIDUAL
      {
         // Test residual
         finest_evaluator_.set_weights(weights_total);
         common::valuesd test_residuals = v.head(n_points_) - finest_evaluator_.evaluate();
         std::cout << "Residual after level " << level << ": " << test_residuals.norm() << std::endl;
      }
#endif
    }

    {
      common::valuesd weights = common::valuesd::Zero(n_points_ + n_poly_basis_);

      // Solve on coarse.
      coarse_->solve(residuals);
      coarse_->set_solution_to(weights);

      if (level < n_fine_levels_ - 1) {
        const auto& coarse_indices = point_idcs_.back();
        common::valuesd coarse_weights(coarse_indices.size() + n_poly_basis_);
        for (size_t i = 0; i < coarse_indices.size(); i++) {
          coarse_weights(i) = weights(coarse_indices[i]);
        }
        coarse_weights.tail(n_poly_basis_) = weights.tail(n_poly_basis_);
        downward_evaluator_[level].set_weights(coarse_weights);

        auto fit = downward_evaluator_[level].evaluate();

        const auto& indices = point_idcs_[level + 1];
        for (size_t i = 0; i < indices.size(); i++) {
          residuals(indices[i]) -= fit(i);
        }
      }

      weights_total += weights;

#if POLATORY_REPORT_RESIDUAL
      {
         // Test residual
         finest_evaluator_.set_weights(weights_total);
         common::valuesd test_residuals = v.head(n_points_) - finest_evaluator_.evaluate();
         std::cout << "Residual after coarse correction: " << test_residuals.norm() << std::endl;
      }
#endif
    }
  }

  return weights_total;
}

size_t ras_preconditioner::size() const {
  return n_points_ + n_poly_basis_;
}

} // namespace preconditioner
} // namespace polatory
