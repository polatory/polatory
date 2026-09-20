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
#include <polatory/interpolation/evaluator.hpp>
#include <polatory/interpolation/symmetric_evaluator.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/preconditioner/binary_cache.hpp>
#include <polatory/preconditioner/coarse_grid.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/preconditioner/domain_divider.hpp>
#include <polatory/preconditioner/fine_grid.hpp>
#include <polatory/preconditioner/mat_q.hpp>
#include <polatory/types.hpp>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class RasPreconditioner : public krylov::LinearOperator {
  static constexpr int kDim = Dim;
  using Bbox = geometry::Bbox<kDim>;
  using CoarseGrid = CoarseGrid<kDim>;
  using Domain = Domain<kDim>;
  using DomainDivider = DomainDivider<kDim>;
  using Evaluator = interpolation::Evaluator<kDim>;
  using FineGrid = FineGrid<kDim>;
  using MatQ = MatQ<kDim>;
  using Model = Model<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Points = geometry::Points<kDim>;
  using SymmetricEvaluator = interpolation::SymmetricEvaluator<kDim>;

  static constexpr double kEvaluatorAccuracy = 0.0;
  static constexpr double kFineToCoarseRatio = 10.0;
  static constexpr Index kNCoarsestPoints = 2048;
  static constexpr bool kReportConditionNumbers = false;
  static constexpr bool kReportResidual = false;

 public:
  RasPreconditioner(const Model& model, const Points& points, const Points& grad_points)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points),
        bbox_(Bbox::from_points(points_).convex_hull(Bbox::from_points(grad_points_))),
        finest_evaluator_(kReportResidual ? std::make_unique<SymmetricEvaluator>(
                                                model, points_, grad_points_, kEvaluatorAccuracy,
                                                kEvaluatorAccuracy)
                                          : nullptr),
        n_levels_(
            std::max(static_cast<int>(std::ceil(
                         std::log(static_cast<double>(mu_ + kDim * sigma_) / kNCoarsestPoints) /
                         std::log(kFineToCoarseRatio))),
                     0) +
            1) {
    point_idcs_.resize(n_levels_);
    grad_point_idcs_.resize(n_levels_);

    std::vector<Index> fixed_point_idcs;
    std::vector<Index> fixed_grad_point_idcs;
    if (l_ > 0) {
      MatQ mat_q(model.poly_degree(), points_, grad_points_);
      if (mat_q.rank() != l_) {
        throw std::runtime_error("the points are not unisolvent");
      }
      for (auto i : mat_q.indices() | std::views::take(l_)) {
        if (i < mu_) {
          fixed_point_idcs.push_back(i);
        } else {
          fixed_grad_point_idcs.push_back((i - mu_) / kDim);
        }
      }
      std::ranges::sort(fixed_point_idcs);
      std::ranges::sort(fixed_grad_point_idcs);
      const auto [first, last] = std::ranges::unique(fixed_grad_point_idcs);
      fixed_grad_point_idcs.erase(first, last);
    }

    {
      auto level = n_levels_ - 1;

      point_idcs_.at(level).resize(mu_);
      std::iota(point_idcs_.at(level).begin(), point_idcs_.at(level).end(), 0);

      grad_point_idcs_.at(level).resize(sigma_);
      std::iota(grad_point_idcs_.at(level).begin(), grad_point_idcs_.at(level).end(), 0);
    }

    const auto& aniso = model_.rbfs().at(0).anisotropy();
    Points a_points;
    Points a_grad_points;
    if (model_.num_rbfs() == 1 && !aniso.isIdentity()) {
      a_points = geometry::transform_points<kDim>(aniso, points_);
      a_grad_points = geometry::transform_points<kDim>(aniso, grad_points_);
    } else {
      a_points = points_;
      a_grad_points = grad_points_;
    }

    fine_grids_.resize(n_levels_);

    std::cout << std::format("{:>8}{:>16}{:>16}{:>16}", "level", "n_domains", "n_points",
                             "n_grad_points");
    if (kReportConditionNumbers) {
      std::cout << std::format("{:>12}{:>12}{:>12}", "cond_min", "cond_med", "cond_max");
    }
    std::cout << std::endl;

    for (auto level = n_levels_ - 1; level >= 1; level--) {
      auto mu = static_cast<Index>(point_idcs_.at(level).size());
      auto sigma = static_cast<Index>(grad_point_idcs_.at(level).size());

      DomainDivider divider(a_points, a_grad_points, point_idcs_.at(level),
                            grad_point_idcs_.at(level));

      auto finest = std::log(mu_ + kDim * sigma_) / std::log(kFineToCoarseRatio);
      auto coarsest = std::log(kNCoarsestPoints) / std::log(kFineToCoarseRatio);
      auto n_coarse_points = static_cast<Index>(std::pow(
          kFineToCoarseRatio, coarsest + (level - 1) * (finest - coarsest) / (n_levels_ - 1)));
      std::tie(point_idcs_.at(level - 1), grad_point_idcs_.at(level - 1)) =
          divider.choose_coarse_points(n_coarse_points, fixed_point_idcs, fixed_grad_point_idcs);

      for (auto& d : std::move(divider).into_domains()) {
        fine_grids_.at(level).emplace_back(model, std::move(d), cache_);
      }

      auto n_grids = static_cast<Index>(fine_grids_.at(level).size());
#pragma omp parallel for schedule(dynamic)
      for (Index i = 0; i < n_grids; i++) {
        auto& fine = fine_grids_.at(level).at(i);
        fine.setup(points_, grad_points_, kReportConditionNumbers);
      }

      std::cout << std::format("{:>8}{:>16}{:>16}{:>16}", level, n_grids, mu, sigma);
      if (kReportConditionNumbers) {
        std::vector<double> conds;
        for (const auto& fine : fine_grids_.at(level)) {
          conds.push_back(fine.condition_number());
        }
        std::ranges::sort(conds);
        std::cout << std::format("{:>12.3e}{:>12.3e}{:>12.3e}", conds.front(),
                                 conds.at(conds.size() / 2), conds.back());
      }
      std::cout << std::endl;
    }

    {
      auto mu = static_cast<Index>(point_idcs_.at(0).size());
      auto sigma = static_cast<Index>(grad_point_idcs_.at(0).size());

      Domain coarse_domain;
      coarse_domain.point_indices = point_idcs_.at(0);
      coarse_domain.grad_point_indices = grad_point_idcs_.at(0);

      coarse_ = std::make_unique<CoarseGrid>(model, std::move(coarse_domain));
      coarse_->setup(points_, grad_points_, kReportConditionNumbers);

      std::cout << std::format("{:>8}{:>16}{:>16}{:>16}", 0, 1, mu, sigma);
      if (kReportConditionNumbers) {
        auto cond = coarse_->condition_number();
        std::cout << std::format("{:>12.3e}{:>12.3e}{:>12.3e}", cond, cond, cond);
      }
      std::cout << std::endl;
    }

    if (n_levels_ == 1) {
      return;
    }

    if (l_ > 0) {
      MonomialBasis poly(model.poly_degree());
      ps_.resize(n_levels_);
      for (auto level = 1; level < n_levels_; level++) {
        const auto& indices = point_idcs_.at(level);
        const auto& grad_indices = grad_point_idcs_.at(level);
        auto mu = static_cast<Index>(indices.size());
        auto sigma = static_cast<Index>(grad_indices.size());

        MatX p = poly.evaluate(points_(indices, kAll), grad_points_(grad_indices, kAll));
        common::orthonormalize_cols(p);

        auto& padded = ps_.at(level);
        padded = MatX::Zero(mu_ + kDim * sigma_, l_);
        padded(indices, kAll) = p.topRows(mu);
        padded.middleRows(mu_, kDim * sigma_)
            .reshaped<Eigen::RowMajor>(sigma_, kDim * l_)(grad_indices, kAll) =
            p.bottomRows(kDim * sigma).reshaped<Eigen::RowMajor>(sigma, kDim * l_);
      }
    }
  }

  VecX operator()(const VecX& v) const override {
    POLATORY_ASSERT(v.rows() == size());

    // v.tail(l_) must be (almost) zero. If that is not the case, the RBF part of the weights
    // was not orthogonalized against the polynomial space in previous iterations.

    VecX residuals = v.head(mu_ + kDim * sigma_);

    if (n_levels_ == 1) {
      return solve(0, residuals);
    }

    VecX weights_total = VecX::Zero(size());
    report_initial_residual(residuals);

    {
      VecX weights = solve(0, residuals);
      update_residuals(0, n_levels_ - 1, weights, residuals);
      weights_total += weights;
      report_residual(0, v, weights_total);
    }

    for (auto level = n_levels_ - 1; level >= 1; level--) {
      {
        VecX weights = solve(level, residuals);
        orthogonalize(level, weights);
        update_residuals(level, level - 1, weights, residuals);
        weights_total += weights;
        report_residual(level, v, weights_total);
      }

      {
        VecX weights = solve(0, residuals);
        if (level > 1) {
          update_residuals(0, level - 1, weights, residuals);
        }
        weights_total += weights;
        report_residual(0, v, weights_total);
      }
    }

    return weights_total;
  }

  Index size() const override { return mu_ + kDim * sigma_ + l_; }

 private:
  Evaluator& evaluator(int src_level, int trg_level) const {
    std::pair key(src_level, trg_level);

    if (!evaluator_.contains(key)) {
      evaluator_.emplace(std::piecewise_construct, std::forward_as_tuple(src_level, trg_level),
                         std::forward_as_tuple(model_, points_(point_idcs_.at(src_level), kAll),
                                               grad_points_(grad_point_idcs_.at(src_level), kAll),
                                               bbox_, kEvaluatorAccuracy, kEvaluatorAccuracy));
      evaluator_.at(key).set_target_points(points_(point_idcs_.at(trg_level), kAll),
                                           grad_points_(grad_point_idcs_.at(trg_level), kAll));
    }

    return evaluator_.at(key);
  }

  void orthogonalize(int level, VecX& weights) const {
    if (l_ > 0) {
      const auto& p = ps_.at(level);
      auto w = weights.head(mu_ + kDim * sigma_);
      w -= p * (p.transpose() * w);
    }
  }

  VecX solve(int level, const VecX& residuals) const {
    VecX weights = VecX::Zero(size());

    if (level == 0) {
      coarse_->solve(residuals);
      coarse_->set_solution_to(weights);
    } else {
      auto n_grids = static_cast<Index>(fine_grids_.at(level).size());
#pragma omp parallel for schedule(dynamic)
      for (Index i = 0; i < n_grids; i++) {
        auto& fine = fine_grids_.at(level).at(i);
        fine.solve(residuals);
        fine.set_solution_to(weights);
      }
    }

    return weights;
  }

  void update_residuals(int src_level, int trg_level, const VecX& weights, VecX& residuals) const {
    const auto& src_indices = point_idcs_.at(src_level);
    const auto& src_grad_indices = grad_point_idcs_.at(src_level);
    auto src_mu = static_cast<Index>(src_indices.size());
    auto src_sigma = static_cast<Index>(src_grad_indices.size());
    VecX src_weights(src_mu + kDim * src_sigma + l_);
    for (Index i = 0; i < src_mu; i++) {
      src_weights(i) = weights(src_indices.at(i));
    }
    for (Index i = 0; i < src_sigma; i++) {
      src_weights.segment<kDim>(src_mu + kDim * i) =
          weights.segment<kDim>(mu_ + kDim * src_grad_indices.at(i));
    }
    src_weights.tail(l_) = weights.tail(l_);
    evaluator(src_level, trg_level).set_weights(src_weights);

    auto fit = evaluator(src_level, trg_level).evaluate();

    const auto& trg_indices = point_idcs_.at(trg_level);
    const auto& trg_grad_indices = grad_point_idcs_.at(trg_level);
    auto trg_mu = static_cast<Index>(trg_indices.size());
    auto trg_sigma = static_cast<Index>(trg_grad_indices.size());
    for (Index i = 0; i < trg_mu; i++) {
      residuals(trg_indices.at(i)) -= fit(i);
    }
    for (Index i = 0; i < trg_sigma; i++) {
      residuals.segment<kDim>(mu_ + kDim * trg_grad_indices.at(i)) -=
          fit.template segment<kDim>(trg_mu + kDim * i);
    }
  }

  void report_initial_residual(const VecX& residuals) const {
    if (kReportResidual) {
      std::cout << std::format("Initial residual: {:f}", residuals.norm()) << std::endl;
    }
  }

  void report_residual(int level, const VecX& v, const VecX& weights_total) const {
    if (kReportResidual) {
      finest_evaluator_->set_weights(weights_total);
      VecX residuals = v.head(mu_ + kDim * sigma_) - finest_evaluator_->evaluate();
      std::cout << std::format("Residual after level {}: {:f}", level, residuals.norm())
                << std::endl;
    }
  }

  const Model& model_;
  const Index l_;
  const Index mu_;
  const Index sigma_;
  const Points points_;
  const Points grad_points_;
  const Bbox bbox_;
  const std::unique_ptr<SymmetricEvaluator> finest_evaluator_;

  MatX lagrange_p_;
  int n_levels_;
  std::vector<std::vector<Index>> point_idcs_;
  std::vector<std::vector<Index>> grad_point_idcs_;
  mutable std::vector<std::vector<FineGrid>> fine_grids_;
  std::unique_ptr<CoarseGrid> coarse_;
  mutable std::map<std::pair<int, int>, Evaluator> evaluator_;
  std::vector<MatX> ps_;
  BinaryCache cache_;
};

}  // namespace polatory::preconditioner
