#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <format>
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
#include <polatory/precision.hpp>
#include <polatory/preconditioner/binary_cache.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/domain_divider.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/types.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class ras_preconditioner : public krylov::linear_operator {
  static constexpr bool kReportResidual = false;
  static constexpr double kCoarseRatio = 0.01;
  static constexpr index_t kNCoarsestPoints = 1024;
  static constexpr int kDim = Dim;

  using Model = model<kDim>;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Domain = domain<kDim>;
  using DomainDivider = domain_divider<kDim>;
  using CoarseGrid = coarse_grid<kDim>;
  using FineGrid = fine_grid<kDim>;
  using Evaluator = interpolation::rbf_evaluator<kDim>;
  using SymmetricEvaluator = interpolation::rbf_symmetric_evaluator<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using LagrangeBasis = polynomial::lagrange_basis<kDim>;
  using UnisolventPointSet = polynomial::unisolvent_point_set<kDim>;

 public:
  ras_preconditioner(const Model& model, const Points& points, const Points& grad_points)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points),
        bbox_(Bbox::from_points(points_).convex_hull(Bbox::from_points(grad_points_))),
        finest_evaluator_(kReportResidual ? std::make_unique<SymmetricEvaluator>(
                                                model, points_, grad_points_, precision::kPrecise)
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
        if (model.poly_degree() == 1 && mu_ == 1 && sigma_ >= 1) {
          // The special case.
          poly_point_idcs = {0};
          LagrangeBasis lagrange_basis(model.poly_degree(), points_, grad_points_.topRows(1));
          lagrange_pt_ = lagrange_basis.evaluate(points_, grad_points_);
        } else {
          // The ordinary case.
          UnisolventPointSet ups(points_, model.poly_degree());
          poly_point_idcs = ups.point_indices();
          LagrangeBasis lagrange_basis(model.poly_degree(), points_(poly_point_idcs, Eigen::all));
          lagrange_pt_ = lagrange_basis.evaluate(points_, grad_points_);
        }

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

      auto aniso = model.rbf()->anisotropy();
      auto divider = std::make_unique<DomainDivider>(
          points_ * aniso.transpose(), grad_points_ * aniso.transpose(), point_idcs_.at(level),
          grad_point_idcs_.at(level), poly_point_idcs);

      auto ratio = level == 1
                       ? static_cast<double>(kNCoarsestPoints) / static_cast<double>(n_mixed_points)
                       : kCoarseRatio;
      std::tie(point_idcs_.at(level - 1), grad_point_idcs_.at(level - 1)) =
          divider->choose_coarse_points(ratio);

      for (auto&& d : divider->into_domains()) {
        fine_grids_.at(level).emplace_back(model, std::move(d), cache_);
      }

      auto n_grids = static_cast<index_t>(fine_grids_.at(level).size());
#pragma omp parallel for
      for (index_t i = 0; i < n_grids; i++) {
        auto& fine = fine_grids_.at(level).at(i);
        fine.setup(points_, grad_points_, lagrange_pt_);
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

    if (l_ > 0) {
      MonomialBasis poly(model.poly_degree());
      p_ = poly.evaluate(points_, grad_points_).transpose();
      common::orthonormalize_cols(p_);

      ap_ = Eigen::MatrixXd(p_.rows(), p_.cols());

      auto finest_evaluator = SymmetricEvaluator(model, points_, grad_points_, precision::kPrecise);
      common::valuesd weights = common::valuesd::Zero(size());
      auto n_cols = p_.cols();
      for (index_t i = 0; i < n_cols; i++) {
        weights.head(mu_ + kDim * sigma_) = p_.col(i);
        finest_evaluator.set_weights(weights);
        ap_.col(i) = finest_evaluator.evaluate();
      }
    }
  }

  common::valuesd operator()(const common::valuesd& v) const override {
    POLATORY_ASSERT(v.rows() == size());

    common::valuesd residuals = v.head(mu_ + kDim * sigma_);

    if (n_levels_ == 1) {
      return solve(0, residuals);
    }

    common::valuesd weights_total = common::valuesd::Zero(size());
    report_initial_residual(residuals);

    {
      common::valuesd weights = solve(0, residuals);
      update_residuals(0, n_levels_ - 1, weights, residuals);
      weights_total += weights;
      report_residual(0, v, weights_total);
    }

    for (auto level = 1; level < n_levels_ - 1; level++) {
      {
        common::valuesd weights = solve(level, residuals);
        update_residuals(level, n_levels_ - 1, weights, residuals);
        weights_total += weights;
        orthogonalize(weights_total, residuals);
        report_residual(level, v, weights_total);
      }

      {
        common::valuesd weights = solve(0, residuals);
        update_residuals(0, n_levels_ - 1, weights, residuals);
        weights_total += weights;
        report_residual(0, v, weights_total);
      }
    }

    for (auto level = n_levels_ - 1; level >= 1; level--) {
      {
        common::valuesd weights = solve(level, residuals);
        update_residuals(level, level - 1, weights, residuals);
        weights_total += weights;
        orthogonalize(weights_total, residuals);
        report_residual(level, v, weights_total);
      }

      {
        common::valuesd weights = solve(0, residuals);
        if (level > 1) {
          update_residuals(0, level - 1, weights, residuals);
        }
        weights_total += weights;
        report_residual(0, v, weights_total);
      }
    }

    return weights_total;
  }

  index_t size() const override { return mu_ + kDim * sigma_ + l_; }

 private:
  Evaluator& evaluator(int src_level, int trg_level) const {
    std::pair key(src_level, trg_level);

    if (!evaluator_.contains(key)) {
      evaluator_.emplace(
          std::piecewise_construct, std::forward_as_tuple(src_level, trg_level),
          std::forward_as_tuple(model_, points_(point_idcs_.at(src_level), Eigen::all),
                                grad_points_(grad_point_idcs_.at(src_level), Eigen::all), bbox_,
                                precision::kFast));
      evaluator_.at(key).set_target_points(
          points_(point_idcs_.at(trg_level), Eigen::all),
          grad_points_(grad_point_idcs_.at(trg_level), Eigen::all));
    }

    return evaluator_.at(key);
  }

  void orthogonalize(common::valuesd& weights, common::valuesd& residuals) const {
    if (l_ > 0) {
      // Orthogonalize weights against P.
      auto n_cols = p_.cols();
      for (index_t i = 0; i < n_cols; i++) {
        auto dot = p_.col(i).dot(weights.head(mu_ + kDim * sigma_));
        weights.head(mu_ + kDim * sigma_) -= dot * p_.col(i);
        residuals += dot * ap_.col(i);
      }
    }
  }

  common::valuesd solve(int level, const common::valuesd& residuals) const {
    common::valuesd weights = common::valuesd::Zero(size());

    if (level == 0) {
      coarse_->solve(residuals);
      coarse_->set_solution_to(weights);
    } else {
      auto n_grids = static_cast<index_t>(fine_grids_.at(level).size());
#pragma omp parallel for schedule(guided)
      for (index_t i = 0; i < n_grids; i++) {
        auto& fine = fine_grids_.at(level).at(i);
        fine.solve(residuals);
        fine.set_solution_to(weights);
      }
    }

    return weights;
  }

  void update_residuals(int src_level, int trg_level, const common::valuesd& weights,
                        common::valuesd& residuals) const {
    const auto& src_indices = point_idcs_.at(src_level);
    const auto& src_grad_indices = grad_point_idcs_.at(src_level);
    auto src_mu = static_cast<index_t>(src_indices.size());
    auto src_sigma = static_cast<index_t>(src_grad_indices.size());
    common::valuesd src_weights(src_mu + kDim * src_sigma + l_);
    for (index_t i = 0; i < src_mu; i++) {
      src_weights(i) = weights(src_indices.at(i));
    }
    for (index_t i = 0; i < src_sigma; i++) {
      src_weights.segment<kDim>(src_mu + kDim * i) =
          weights.segment<kDim>(mu_ + kDim * src_grad_indices.at(i));
    }
    src_weights.tail(l_) = weights.tail(l_);
    evaluator(src_level, trg_level).set_weights(src_weights);

    auto fit = evaluator(src_level, trg_level).evaluate();

    const auto& trg_indices = point_idcs_.at(trg_level);
    const auto& trg_grad_indices = grad_point_idcs_.at(trg_level);
    auto trg_mu = static_cast<index_t>(trg_indices.size());
    auto trg_sigma = static_cast<index_t>(trg_grad_indices.size());
    for (index_t i = 0; i < trg_mu; i++) {
      residuals(trg_indices.at(i)) -= fit(i);
    }
    for (index_t i = 0; i < trg_sigma; i++) {
      residuals.segment<kDim>(mu_ + kDim * trg_grad_indices.at(i)) -=
          fit.template segment<kDim>(trg_mu + kDim * i);
    }
  }

  void report_initial_residual(const common::valuesd& residuals) const {
    if (kReportResidual) {
      std::cout << std::format("Initial residual: {:f}", residuals.norm()) << std::endl;
    }
  }

  void report_residual(int level, const common::valuesd& v,
                       const common::valuesd& weights_total) const {
    if (kReportResidual) {
      finest_evaluator_->set_weights(weights_total);
      common::valuesd residuals = v.head(mu_ + kDim * sigma_) - finest_evaluator_->evaluate();
      std::cout << std::format("Residual after level {}: {:f}", level, residuals.norm())
                << std::endl;
    }
  }

  const Model& model_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const Points points_;
  const Points grad_points_;
  const Bbox bbox_;
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
  binary_cache cache_;
};

}  // namespace polatory::preconditioner
