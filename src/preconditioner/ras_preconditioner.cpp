#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <polatory/common/macros.hpp>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/unisolvent_point_set.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/domain_divider.hpp>
#include <polatory/preconditioner/ras_preconditioner.hpp>
#include <tuple>
#include <utility>

namespace polatory::preconditioner {

ras_preconditioner::ras_preconditioner(const model& model, const geometry::points3d& points,
                                       const geometry::points3d& grad_points)
    : model_without_poly_(model.without_poly()),
      dim_(model.poly_dimension()),
      l_(model.poly_basis_size()),
      mu_(points.rows()),
      sigma_(grad_points.rows()),
      points_(points),
      grad_points_(grad_points),
      finest_evaluator_(kReportResidual
                            ? std::make_unique<interpolation::rbf_symmetric_evaluator<Order>>(
                                  model, points_, grad_points_)
                            : nullptr) {
  auto n_fine_levels =
      std::max(0, static_cast<int>(std::ceil(std::log(static_cast<double>(mu_ + sigma_) /
                                                      static_cast<double>(n_coarsest_points)) /
                                             log(1.0 / coarse_ratio))));
  n_levels_ = n_fine_levels + 1;

  point_idcs_.resize(n_levels_);
  grad_point_idcs_.resize(n_levels_);

  std::vector<index_t> poly_point_idcs;
  {
    auto level = n_levels_ - 1;

    if (l_ > 0) {
      polynomial::unisolvent_point_set ups(points_, model.poly_dimension(), model.poly_degree());

      poly_point_idcs = ups.point_indices();
      polynomial::lagrange_basis lagrange_basis(model.poly_dimension(), model.poly_degree(),
                                                points_(poly_point_idcs, Eigen::all));
      lagrange_pt_ = lagrange_basis.evaluate(points_, grad_points_);

      point_idcs_.at(level) = poly_point_idcs;
      point_idcs_.at(level).reserve(mu_);
      for (index_t i = 0; i < mu_; i++) {
        if (!std::binary_search(poly_point_idcs.begin(), poly_point_idcs.end(), i)) {
          point_idcs_.at(level).push_back(i);
        }
      }
    } else {
      point_idcs_.at(level).resize(mu_);
      std::iota(point_idcs_.at(level).begin(), point_idcs_.at(level).end(), 0);
    }

    grad_point_idcs_.at(level).resize(sigma_);
    std::iota(grad_point_idcs_.at(level).begin(), grad_point_idcs_.at(level).end(), 0);
  }

  fine_grids_.resize(n_levels_);

  std::cout << std::setw(8) << "level" << std::setw(16) << "n_domains" << std::setw(16)
            << "n_points" << std::endl;

  for (auto level = n_levels_ - 1; level >= 1; level--) {
    auto n_mixed_points =
        static_cast<index_t>(point_idcs_.at(level).size() + grad_point_idcs_.at(level).size());

    auto divider = std::make_unique<domain_divider>(points_, grad_points_, point_idcs_.at(level),
                                                    grad_point_idcs_.at(level), poly_point_idcs);

    auto ratio = level == 1
                     ? static_cast<double>(n_coarsest_points) / static_cast<double>(n_mixed_points)
                     : coarse_ratio;
    std::tie(point_idcs_.at(level - 1), grad_point_idcs_.at(level - 1)) =
        divider->choose_coarse_points(ratio);

    for (auto&& d : divider->into_domains()) {
      fine_grids_.at(level).emplace_back(model, std::move(d));
    }

    auto n_grids = static_cast<index_t>(fine_grids_.at(level).size());
    if (!kRecomputeAndClear) {
#pragma omp parallel for
      for (index_t i = 0; i < n_grids; i++) {
        auto& fine = fine_grids_.at(level).at(i);
        fine.setup(points_, grad_points_, lagrange_pt_);
      }
    }

    std::cout << std::setw(8) << level << std::setw(16) << n_grids << std::setw(16)
              << n_mixed_points << std::endl;
  }

  {
    auto n_mixed_points =
        static_cast<index_t>(point_idcs_.at(0).size() + grad_point_idcs_.at(0).size());

    domain coarse_domain;
    coarse_domain.point_indices = point_idcs_.at(0);
    coarse_domain.grad_point_indices = grad_point_idcs_.at(0);

    coarse_ = std::make_unique<coarse_grid>(model, std::move(coarse_domain));
    coarse_->setup(points_, grad_points_, lagrange_pt_);

    std::cout << std::setw(8) << 0 << std::setw(16) << 1 << std::setw(16) << n_mixed_points
              << std::endl;
  }

  if (n_levels_ == 1) {
    return;
  }

  auto bbox = geometry::bbox3d::from_points(points_).convex_hull(
      geometry::bbox3d::from_points(grad_points_));
  for (auto level = 1; level < n_levels_; level++) {
    if (level == n_levels_ - 1) {
      add_evaluator(level, level - 1, model_without_poly_, points_, grad_points_, bbox);
    } else {
      add_evaluator(level, level - 1, model_without_poly_,
                    points_(point_idcs_.at(level), Eigen::all),
                    grad_points_(grad_point_idcs_.at(level), Eigen::all), bbox);
    }
    evaluator(level, level - 1)
        .set_field_points(points_(point_idcs_.at(level - 1), Eigen::all),
                          grad_points_(grad_point_idcs_.at(level - 1), Eigen::all));
  }

  for (auto level = 1; level < n_levels_ - 1; level++) {
    add_evaluator(0, level, model, points_(point_idcs_.at(0), Eigen::all),
                  grad_points_(grad_point_idcs_.at(0), Eigen::all), bbox);
    evaluator(0, level).set_field_points(points_(point_idcs_.at(level), Eigen::all),
                                         grad_points_(grad_point_idcs_.at(level), Eigen::all));
  }

  if (l_ > 0) {
    polynomial::monomial_basis poly(model.poly_dimension(), model.poly_degree());
    p_ = poly.evaluate(points_, grad_points_).transpose();
    common::orthonormalize_cols(p_);

    ap_ = Eigen::MatrixXd(p_.rows(), p_.cols());

    auto finest_evaluator =
        interpolation::rbf_symmetric_evaluator<Order>(model_without_poly_, points_, grad_points_);
    auto n_cols = p_.cols();
    for (index_t i = 0; i < n_cols; i++) {
      finest_evaluator.set_weights(p_.col(i));
      ap_.col(i) = finest_evaluator.evaluate();
    }
  }
}

common::valuesd ras_preconditioner::operator()(const common::valuesd& v) const {
  POLATORY_ASSERT(v.rows() == size());

  common::valuesd residuals = v.head(mu_ + dim_ * sigma_);
  common::valuesd weights_total = common::valuesd::Zero(size());
  if (n_levels_ == 1) {
    coarse_->solve(residuals);
    coarse_->set_solution_to(weights_total);
    return weights_total;
  }

  if (kReportResidual) {
    std::cout << "Initial residual: " << residuals.norm() << std::endl;
  }

  for (auto level = n_levels_ - 1; level >= 1; level--) {
    {
      common::valuesd weights = common::valuesd::Zero(mu_ + dim_ * sigma_);

      // Solve on level `level`.
      auto n_grids = static_cast<index_t>(fine_grids_.at(level).size());
#pragma omp parallel for schedule(guided)
      for (index_t i = 0; i < n_grids; i++) {
        auto& fine = fine_grids_.at(level).at(i);
        if (kRecomputeAndClear) {
          fine.setup(points_, grad_points_, lagrange_pt_);
        }
        fine.solve(residuals);
        fine.set_solution_to(weights);
        if (kRecomputeAndClear) {
          fine.clear();
        }
      }

      // Evaluate residuals on level `level` - 1.
      if (level < n_levels_ - 1) {
        const auto& finer_indices = point_idcs_.at(level);
        const auto& finer_grad_indices = grad_point_idcs_.at(level);
        auto finer_mu = static_cast<index_t>(finer_indices.size());
        auto finer_sigma = static_cast<index_t>(finer_grad_indices.size());
        common::valuesd finer_weights(finer_mu + dim_ * finer_sigma);
        for (index_t i = 0; i < finer_mu; i++) {
          finer_weights(i) = weights(finer_indices.at(i));
        }
        for (index_t i = 0; i < finer_sigma; i++) {
          finer_weights.segment(finer_mu + dim_ * i, dim_) =
              weights.segment(mu_ + dim_ * finer_grad_indices.at(i), dim_);
        }
        evaluator(level, level - 1).set_weights(finer_weights);
      } else {
        evaluator(level, level - 1).set_weights(weights);
      }
      auto fit = evaluator(level, level - 1).evaluate();

      const auto& indices = point_idcs_.at(level - 1);
      const auto& grad_indices = grad_point_idcs_.at(level - 1);
      auto mu = static_cast<index_t>(indices.size());
      auto sigma = static_cast<index_t>(grad_indices.size());
      for (index_t i = 0; i < mu; i++) {
        residuals(indices.at(i)) -= fit(i);
      }
      for (index_t i = 0; i < sigma; i++) {
        residuals.segment(mu_ + dim_ * grad_indices.at(i), dim_) -=
            fit.segment(mu + dim_ * i, dim_);
      }

      if (l_ > 0) {
        // Orthogonalize weights against P.
        auto n_cols = p_.cols();
        for (index_t i = 0; i < n_cols; i++) {
          auto dot = p_.col(i).dot(weights);
          weights -= dot * p_.col(i);
          residuals += dot * ap_.col(i);
        }
      }

      weights_total.head(mu_ + dim_ * sigma_) += weights;

      if (kReportResidual) {
        finest_evaluator_->set_weights(weights_total);
        common::valuesd test_residuals =
            v.head(mu_ + dim_ * sigma_) - finest_evaluator_->evaluate();
        std::cout << "Residual after level " << level << ": " << test_residuals.norm() << std::endl;
      }
    }

    {
      common::valuesd weights = common::valuesd::Zero(size());

      // Solve on level 0.
      coarse_->solve(residuals);
      coarse_->set_solution_to(weights);

      // Update residuals on level `level` - 1.
      if (level > 1) {
        const auto& coarse_indices = point_idcs_.at(0);
        const auto& coarse_grad_indices = grad_point_idcs_.at(0);
        auto coarse_mu = static_cast<index_t>(coarse_indices.size());
        auto coarse_sigma = static_cast<index_t>(coarse_grad_indices.size());
        common::valuesd coarse_weights(coarse_mu + dim_ * coarse_sigma + l_);
        for (index_t i = 0; i < coarse_mu; i++) {
          coarse_weights(i) = weights(coarse_indices.at(i));
        }
        for (index_t i = 0; i < coarse_sigma; i++) {
          coarse_weights.segment(coarse_mu + dim_ * i, dim_) =
              weights.segment(mu_ + dim_ * coarse_grad_indices.at(i), dim_);
        }
        coarse_weights.tail(l_) = weights.tail(l_);
        evaluator(0, level - 1).set_weights(coarse_weights);

        auto fit = evaluator(0, level - 1).evaluate();

        const auto& indices = point_idcs_.at(level - 1);
        const auto& grad_indices = grad_point_idcs_.at(level - 1);
        auto mu = static_cast<index_t>(indices.size());
        auto sigma = static_cast<index_t>(grad_indices.size());
        for (index_t i = 0; i < mu; i++) {
          residuals(indices.at(i)) -= fit(i);
        }
        for (index_t i = 0; i < sigma; i++) {
          residuals.segment(mu_ + dim_ * grad_indices.at(i), dim_) -=
              fit.segment(mu + dim_ * i, dim_);
        }
      }

      weights_total += weights;

      if (kReportResidual) {
        finest_evaluator_->set_weights(weights_total);
        common::valuesd test_residuals =
            v.head(mu_ + dim_ * sigma_) - finest_evaluator_->evaluate();
        std::cout << "Residual after level 0: " << test_residuals.norm() << std::endl;
      }
    }
  }

  return weights_total;
}

index_t ras_preconditioner::size() const { return mu_ + dim_ * sigma_ + l_; }

}  // namespace polatory::preconditioner
