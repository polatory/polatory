#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <polatory/common/macros.hpp>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_symmetric_evaluator.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/unisolvent_point_set.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/domain_divider.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/types.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

template <class Model>
class ras_preconditioner : public krylov::linear_operator {
  static constexpr bool kRecomputeAndClear = true;
  static constexpr bool kReportResidual = false;
  static constexpr double kCoarseRatio = 0.125;
  static constexpr index_t kNCoarsestPoints = 1024;
  static constexpr int kDim = Model::kDim;

  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Domain = domain<kDim>;
  using DomainDivider = domain_divider<kDim>;
  using CoarseGrid = coarse_grid<Model>;
  using FineGrid = fine_grid<Model>;
  using Evaluator = interpolation::rbf_evaluator<Model>;
  using SymmetricEvaluator = interpolation::rbf_symmetric_evaluator<Model>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using LagrangeBasis = polynomial::lagrange_basis<kDim>;

 public:
  ras_preconditioner(const Model& model, const Points& points, const Points& grad_points)
      : model_without_poly_(model.without_poly()),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points),
        finest_evaluator_(kReportResidual ? std::make_unique<SymmetricEvaluator>(
                                                model, points_, grad_points_, precision::kFast)
                                          : nullptr) {
    auto n_fine_levels =
        std::max(0, static_cast<int>(std::ceil(std::log(static_cast<double>(mu_ + sigma_) /
                                                        static_cast<double>(kNCoarsestPoints)) /
                                               log(1.0 / kCoarseRatio))));
    n_levels_ = n_fine_levels + 1;

    point_idcs_.resize(n_levels_);
    grad_point_idcs_.resize(n_levels_);

    std::vector<index_t> poly_point_idcs;
    {
      auto level = n_levels_ - 1;

      if (l_ > 0) {
        polynomial::unisolvent_point_set<kDim> ups(points_, model.poly_degree());

        poly_point_idcs = ups.point_indices();
        LagrangeBasis lagrange_basis(model.poly_degree(), points_(poly_point_idcs, Eigen::all));
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

      auto aniso = model.rbf().anisotropy();
      auto divider = std::make_unique<DomainDivider>(
          points_ * aniso.transpose(), grad_points_ * aniso.transpose(), point_idcs_.at(level),
          grad_point_idcs_.at(level), poly_point_idcs);

      auto ratio = level == 1
                       ? static_cast<double>(kNCoarsestPoints) / static_cast<double>(n_mixed_points)
                       : kCoarseRatio;
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

      Domain coarse_domain;
      coarse_domain.point_indices = point_idcs_.at(0);
      coarse_domain.grad_point_indices = grad_point_idcs_.at(0);

      coarse_ = std::make_unique<CoarseGrid>(model, std::move(coarse_domain));
      coarse_->setup(points_, grad_points_, lagrange_pt_);

      std::cout << std::setw(8) << 0 << std::setw(16) << 1 << std::setw(16) << n_mixed_points
                << std::endl;
    }

    if (n_levels_ == 1) {
      return;
    }

    auto bbox = Bbox::from_points(points_).convex_hull(Bbox::from_points(grad_points_));
    for (auto level = 1; level < n_levels_; level++) {
      if (level == n_levels_ - 1) {
        add_evaluator(level, level - 1, model_without_poly_, points_, grad_points_, bbox,
                      precision::kFast);
      } else {
        add_evaluator(level, level - 1, model_without_poly_,
                      points_(point_idcs_.at(level), Eigen::all),
                      grad_points_(grad_point_idcs_.at(level), Eigen::all), bbox, precision::kFast);
      }
      evaluator(level, level - 1)
          .set_target_points(points_(point_idcs_.at(level - 1), Eigen::all),
                             grad_points_(grad_point_idcs_.at(level - 1), Eigen::all));
    }

    for (auto level = 1; level < n_levels_ - 1; level++) {
      add_evaluator(0, level, model, points_(point_idcs_.at(0), Eigen::all),
                    grad_points_(grad_point_idcs_.at(0), Eigen::all), bbox, precision::kFast);
      evaluator(0, level).set_target_points(points_(point_idcs_.at(level), Eigen::all),
                                            grad_points_(grad_point_idcs_.at(level), Eigen::all));
    }

    if (l_ > 0) {
      MonomialBasis poly(model.poly_degree());
      p_ = poly.evaluate(points_, grad_points_).transpose();
      common::orthonormalize_cols(p_);

      ap_ = Eigen::MatrixXd(p_.rows(), p_.cols());

      auto finest_evaluator =
          SymmetricEvaluator(model_without_poly_, points_, grad_points_, precision::kFast);
      auto n_cols = p_.cols();
      for (index_t i = 0; i < n_cols; i++) {
        finest_evaluator.set_weights(p_.col(i));
        ap_.col(i) = finest_evaluator.evaluate();
      }
    }
  }

  common::valuesd operator()(const common::valuesd& v) const override {
    POLATORY_ASSERT(v.rows() == size());

    common::valuesd residuals = v.head(mu_ + kDim * sigma_);
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
        common::valuesd weights = common::valuesd::Zero(mu_ + kDim * sigma_);

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
          common::valuesd finer_weights(finer_mu + kDim * finer_sigma);
          for (index_t i = 0; i < finer_mu; i++) {
            finer_weights(i) = weights(finer_indices.at(i));
          }
          for (index_t i = 0; i < finer_sigma; i++) {
            finer_weights.segment(finer_mu + kDim * i, kDim) =
                weights.segment(mu_ + kDim * finer_grad_indices.at(i), kDim);
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
          residuals.segment(mu_ + kDim * grad_indices.at(i), kDim) -=
              fit.segment(mu + kDim * i, kDim);
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

        weights_total.head(mu_ + kDim * sigma_) += weights;

        if (kReportResidual) {
          finest_evaluator_->set_weights(weights_total);
          common::valuesd test_residuals =
              v.head(mu_ + kDim * sigma_) - finest_evaluator_->evaluate();
          std::cout << "Residual after level " << level << ": " << test_residuals.norm()
                    << std::endl;
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
          common::valuesd coarse_weights(coarse_mu + kDim * coarse_sigma + l_);
          for (index_t i = 0; i < coarse_mu; i++) {
            coarse_weights(i) = weights(coarse_indices.at(i));
          }
          for (index_t i = 0; i < coarse_sigma; i++) {
            coarse_weights.segment(coarse_mu + kDim * i, kDim) =
                weights.segment(mu_ + kDim * coarse_grad_indices.at(i), kDim);
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
            residuals.segment(mu_ + kDim * grad_indices.at(i), kDim) -=
                fit.segment(mu + kDim * i, kDim);
          }
        }

        weights_total += weights;

        if (kReportResidual) {
          finest_evaluator_->set_weights(weights_total);
          common::valuesd test_residuals =
              v.head(mu_ + kDim * sigma_) - finest_evaluator_->evaluate();
          std::cout << "Residual after level 0: " << test_residuals.norm() << std::endl;
        }
      }
    }

    return weights_total;
  }

  index_t size() const override { return mu_ + kDim * sigma_ + l_; }

 private:
  template <class... Args>
  void add_evaluator(int from_level, int to_level, Args&&... args) {
    evaluator_.emplace(std::piecewise_construct, std::forward_as_tuple(from_level, to_level),
                       std::forward_as_tuple(std::forward<Args>(args)...));
  }

  Evaluator& evaluator(int from_level, int to_level) const {
    return evaluator_.at({from_level, to_level});
  }

  const Model model_without_poly_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const Points points_;
  const Points grad_points_;
  const std::unique_ptr<SymmetricEvaluator> finest_evaluator_;

  Eigen::MatrixXd lagrange_pt_;
  int n_levels_;
  std::vector<std::vector<index_t>> point_idcs_;
  std::vector<std::vector<index_t>> grad_point_idcs_;
  mutable std::vector<std::vector<FineGrid>> fine_grids_;
  std::unique_ptr<CoarseGrid> coarse_;
  mutable std::map<std::pair<int, int>, Evaluator> evaluator_;
  Eigen::MatrixXd p_;
  Eigen::MatrixXd ap_;
};

}  // namespace polatory::preconditioner
