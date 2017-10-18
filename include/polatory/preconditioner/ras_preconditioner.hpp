// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#define REPORT_RESIDUAL 0
#define CLEAR_AND_RECOMPUTE 1

#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "polatory/common/bsearch.hpp"
#include "polatory/common/vector_view.hpp"
#include "polatory/interpolation/rbf_evaluator.hpp"
#include "polatory/interpolation/rbf_symmetric_evaluator.hpp"
#include "polatory/krylov/linear_operator.hpp"
#include "polatory/polynomial/basis_base.hpp"
#include "polatory/polynomial/lagrange_basis.hpp"
#include "polatory/polynomial/orthonormal_basis.hpp"
#include "polatory/preconditioner/coarse_grid.hpp"
#include "polatory/preconditioner/domain_divider.hpp"
#include "polatory/preconditioner/fine_grid.hpp"
#include "polatory/rbf/rbf_base.hpp"

namespace polatory {
namespace preconditioner {

template <class Floating>
class ras_preconditioner : public krylov::linear_operator {
  using LagrangeBasis = polynomial::lagrange_basis<Floating>;
  using FineGrid = fine_grid<Floating>;
  using CoarseGrid = coarse_grid<Floating>;

  static constexpr int Order = 6;
  const double coarse_ratio = 0.125;
  const size_t n_coarsest_points = 1024;
  const int poly_degree;
  const std::vector<Eigen::Vector3d> points;
  const size_t n_points;
  const size_t n_polynomials;
  int n_fine_levels;

#if REPORT_RESIDUAL
  mutable interpolation::rbf_symmetric_evaluator<Order> finest_evaluator;
#endif

  std::vector<std::vector<size_t>> point_idcs_;
  std::vector<size_t> poly_point_idcs_;
  mutable std::vector<std::vector<FineGrid>> fine_grids;
  std::shared_ptr<LagrangeBasis> lagrange_basis_;
  std::unique_ptr<CoarseGrid> coarse;
  std::vector<interpolation::rbf_evaluator<Order>> downward_evaluator;
  std::vector<interpolation::rbf_evaluator<Order>> upward_evaluator;
  Eigen::MatrixXd p;
  Eigen::MatrixXd ap;

public:
  template <typename Container>
  ras_preconditioner(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                     const Container& in_points)
    : poly_degree(poly_degree)
    , points(in_points.begin(), in_points.end())
    , n_points(in_points.size())
    , n_polynomials(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
#if REPORT_RESIDUAL
    , finest_evaluator(rbf, poly_dimension, poly_degree, points)
#endif
  {
    point_idcs_.push_back(std::vector<size_t>(n_points));
    std::iota(point_idcs_.back().begin(), point_idcs_.back().end(), 0);

    if (poly_degree >= 0) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<int> dist(0, n_points - 1);
      std::set<size_t> poly_point_idcs;

      while (poly_point_idcs.size() < n_polynomials) {
        size_t idx = dist(gen);
        if (!poly_point_idcs.insert(idx).second)
          continue;

        auto it = common::bsearch_eq(point_idcs_.back().begin(), point_idcs_.back().end(), idx);
        point_idcs_.back().erase(it);
      }

      poly_point_idcs_ = std::vector<size_t>(poly_point_idcs.begin(), poly_point_idcs.end());
      point_idcs_.back().insert(point_idcs_.back().begin(), poly_point_idcs_.begin(), poly_point_idcs_.end());
      lagrange_basis_ = std::make_shared<LagrangeBasis>(poly_dimension, poly_degree, common::make_view(points, poly_point_idcs_));
    }

    n_fine_levels = std::max(0, int(
      std::ceil(std::log(double(n_points) / double(n_coarsest_points)) / log(1.0 / coarse_ratio))));
    if (n_fine_levels == 0) {
      coarse = std::make_unique<CoarseGrid>(rbf, lagrange_basis_, point_idcs_.back(), points);
      return;
    }

    auto divider = std::make_unique<domain_divider>(points, point_idcs_.back(), poly_point_idcs_);

    fine_grids.push_back(std::vector<FineGrid>());
    for (const auto& d : divider->domains()) {
      fine_grids.back().push_back(FineGrid(rbf, lagrange_basis_, d.point_indices, d.inner_point));
    }
#if !CLEAR_AND_RECOMPUTE
#pragma omp parallel for
    for (size_t i = 0; i < fine_grids.back().size(); i++) {
       auto& fine = fine_grids.back()[i];
       fine.setup(points);
    }
#endif
    std::cout << "Number of points in level 0: " << points.size() << std::endl;
    std::cout << "Number of domains in level 0: " << fine_grids.back().size() << std::endl;

    auto ratio = 0 == n_fine_levels - 1
                 ? double(n_coarsest_points) / double(points.size())
                 : coarse_ratio;
    upward_evaluator.push_back(interpolation::rbf_evaluator<Order>(rbf, -1, -1, points));
    point_idcs_.push_back(divider->choose_coarse_points(ratio));
    upward_evaluator.back().set_field_points(common::make_view(points, point_idcs_.back()));

    for (int level = 1; level < n_fine_levels; level++) {
      divider = std::make_unique<domain_divider>(points, point_idcs_.back(), poly_point_idcs_);

      fine_grids.push_back(std::vector<FineGrid>());
      for (const auto& d : divider->domains()) {
        fine_grids.back().push_back(FineGrid(rbf, lagrange_basis_, d.point_indices, d.inner_point));
      }
#if !CLEAR_AND_RECOMPUTE
#pragma omp parallel for
      for (size_t i = 0; i < fine_grids.back().size(); i++) {
         auto& fine = fine_grids.back()[i];
         fine.setup(points);
      }
#endif
      std::cout << "Number of points in level " << level << ": " << point_idcs_.back().size() << std::endl;
      std::cout << "Number of domains in level " << level << ": " << fine_grids.back().size() << std::endl;

      ratio = level == n_fine_levels - 1
              ? double(n_coarsest_points) / double(point_idcs_.back().size())
              : coarse_ratio;
      upward_evaluator.push_back(
        interpolation::rbf_evaluator<Order>(rbf, -1, -1, common::make_view(points, point_idcs_.back())));
      point_idcs_.push_back(divider->choose_coarse_points(ratio));
      upward_evaluator.back().set_field_points(common::make_view(points, point_idcs_.back()));
    }

    std::cout << "Number of points in coarse: " << point_idcs_.back().size() << std::endl;
    coarse = std::make_unique<CoarseGrid>(rbf, lagrange_basis_, point_idcs_.back(), points);

    for (int level = 1; level < n_fine_levels; level++) {
      downward_evaluator.push_back(
        interpolation::rbf_evaluator<Order>(rbf, poly_dimension, poly_degree, common::make_view(points, point_idcs_.back())));
      downward_evaluator.back().set_field_points(common::make_view(points, point_idcs_[level]));
    }

    if (poly_degree >= 0) {
      polynomial::orthonormal_basis<> poly(poly_dimension, poly_degree, points);
      p = poly.evaluate_points(points).transpose();
      ap = Eigen::MatrixXd(p.rows(), p.cols());

      auto finest_evaluator = interpolation::rbf_symmetric_evaluator<Order>(rbf, -1, -1, points);
      for (size_t i = 0; i < p.cols(); i++) {
        finest_evaluator.set_weights(p.col(i));
        ap.col(i) = finest_evaluator.evaluate();
      }
    }
  }

  Eigen::VectorXd operator()(const Eigen::VectorXd& v) const override {
    assert(v.size() == size());

    Eigen::VectorXd residuals = v.head(n_points);
    Eigen::VectorXd weights_total = Eigen::VectorXd::Zero(size());
    if (n_fine_levels == 0) {
      coarse->solve(residuals);
      coarse->set_solution_to(weights_total);
      return weights_total;
    }

#if REPORT_RESIDUAL
    std::cout << "Initial residual: " << residuals.norm() << std::endl;
#endif

    for (int level = 0; level < n_fine_levels; level++) {
      {
        std::cout << "Start of level " << level << std::endl;

        Eigen::VectorXd weights = Eigen::VectorXd::Zero(n_points);

        // Solve on subdomains.
#pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < fine_grids[level].size(); i++) {
          auto& fine = fine_grids[level][i];
#if CLEAR_AND_RECOMPUTE
          fine.setup(points);
#endif
          fine.solve(residuals);
          fine.set_solution_to(weights);
#if CLEAR_AND_RECOMPUTE
          fine.clear();
#endif
        }

        // Evaluate residuals at coarse points.
        if (level > 0) {
          const auto& finer_indices = point_idcs_[level];
          Eigen::VectorXd finer_weights(finer_indices.size());
          for (size_t i = 0; i < finer_indices.size(); i++) {
            finer_weights(i) = weights(finer_indices[i]);
          }
          upward_evaluator[level].set_weights(finer_weights);
        } else {
          upward_evaluator[level].set_weights(weights);
        }
        auto fit = upward_evaluator[level].evaluate();

        const auto& indices = point_idcs_[level + 1];
        for (size_t i = 0; i < indices.size(); i++) {
          residuals(indices[i]) -= fit(i);
        }

        if (poly_degree >= 0) {
          // Orthogonalize weights against P.
          for (size_t i = 0; i < p.cols(); i++) {
            auto dot = p.col(i).dot(weights);
            weights -= dot * p.col(i);
            residuals -= dot * ap.col(i);
          }
        }

        weights_total.head(n_points) += weights;

        std::cout << "End of level " << level << std::endl;

#if REPORT_RESIDUAL
        {
           // Test residual
           finest_evaluator.set_weights(weights_total);
           Eigen::VectorXd test_residuals = v.head(n_points) - finest_evaluator.evaluate();
           std::cout << "Residual after level " << level << ": " << test_residuals.norm() << std::endl;
        }
#endif
      }

      {
        std::cout << "Start of coarse correction" << std::endl;

        Eigen::VectorXd weights = Eigen::VectorXd::Zero(n_points + n_polynomials);

        // Solve on coarse.
        coarse->solve(residuals);
        coarse->set_solution_to(weights);

        if (level < n_fine_levels - 1) {
          const auto& coarse_indices = point_idcs_.back();
          Eigen::VectorXd coarse_weights(coarse_indices.size() + n_polynomials);
          for (size_t i = 0; i < coarse_indices.size(); i++) {
            coarse_weights(i) = weights(coarse_indices[i]);
          }
          coarse_weights.tail(n_polynomials) = weights.tail(n_polynomials);
          downward_evaluator[level].set_weights(coarse_weights);

          auto fit = downward_evaluator[level].evaluate();

          const auto& indices = point_idcs_[level + 1];
          for (size_t i = 0; i < indices.size(); i++) {
            residuals(indices[i]) -= fit(i);
          }
        }

        weights_total += weights;

        std::cout << "End of coarse correction" << std::endl;

#if REPORT_RESIDUAL
        {
           // Test residual
           finest_evaluator.set_weights(weights_total);
           Eigen::VectorXd test_residuals = v.head(n_points) - finest_evaluator.evaluate();
           std::cout << "Residual after coarse correction: " << test_residuals.norm() << std::endl;
        }
#endif
      }
    }

    return weights_total;
  }

  size_t size() const override {
    return n_points + n_polynomials;
  }
};

} // namespace preconditioner
} // namespace polatory
