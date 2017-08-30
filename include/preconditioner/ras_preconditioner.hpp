// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#define REPORT_RESIDUAL 0
#define CLEAR_AND_RECOMPUTE 1

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "coarse_grid.hpp"
#include "../common/vector_view.hpp"
#include "domain_divider.hpp"
#include "fine_grid.hpp"
#include "../interpolation/rbf_evaluator.hpp"
#include "../interpolation/rbf_symmetric_evaluator.hpp"
#include "../krylov/linear_operator.hpp"
#include "../polynomial.hpp"
#include "../rbf/rbf_base.hpp"

namespace polatory {
namespace preconditioner {

struct ras_preconditioner : krylov::linear_operator {
private:
   using Float = float;
   using FineGrid = fine_grid<Float>;
   using CoarseGrid = coarse_grid<Float>;

   static constexpr int Order = 6;
   const double coarse_ratio = 0.125;
   const size_t n_coarsest_points = 1024;
   const int poly_degree;
   const std::vector<Eigen::Vector3d> points;
   const size_t n_points;
   const size_t n_polynomials;
   int n_fine_levels;

#if REPORT_RESIDUAL
   interpolation::rbf_symmetric_evaluator<Order> finest_evaluator;
#endif

   mutable std::vector<std::vector<FineGrid>> fine_grids;
   std::vector<std::vector<size_t>> point_indices;
   std::unique_ptr<CoarseGrid> coarse;
   std::vector<interpolation::rbf_evaluator<Order>> downward_evaluator;
   std::vector<interpolation::rbf_evaluator<Order>> upward_evaluator;
   Eigen::MatrixXd p;
   Eigen::MatrixXd ap;

public:
   template<typename Container>
   ras_preconditioner(const rbf::rbf_base& rbf, int poly_degree,
      const Container& in_points)
      : poly_degree(poly_degree)
      , points(in_points.begin(), in_points.end())
      , n_points(in_points.size())
      , n_polynomials(polynomial::basis_base::dimension(poly_degree))
#if REPORT_RESIDUAL
      , finest_evaluator(rbf, poly_degree, points)
#endif
   {
      n_fine_levels = std::max(0, int(std::ceil(std::log(double(n_points) / double(n_coarsest_points)) / log(1.0 / coarse_ratio))));
      if (n_fine_levels == 0)
         return;

      auto divider = std::make_unique<domain_divider>(points);

      fine_grids.push_back(std::vector<FineGrid>());
      for (const auto& d : divider->domains()) {
         fine_grids.back().push_back(FineGrid(rbf, d.point_indices, d.inner_point));
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
      upward_evaluator.push_back(interpolation::rbf_evaluator<Order>(rbf, -1, points));
      point_indices.push_back(divider->choose_coarse_points(ratio));
      upward_evaluator.back().set_field_points(common::make_view(points, point_indices.back()));

      for (int level = 1; level < n_fine_levels; level++) {
         divider = std::make_unique<domain_divider>(points, point_indices.back());

         fine_grids.push_back(std::vector<FineGrid>());
         for (const auto& d : divider->domains()) {
            fine_grids.back().push_back(FineGrid(rbf, d.point_indices, d.inner_point));
         }
#if !CLEAR_AND_RECOMPUTE
#pragma omp parallel for
         for (size_t i = 0; i < fine_grids.back().size(); i++) {
            auto& fine = fine_grids.back()[i];
            fine.setup(points);
         }
#endif
         std::cout << "Number of points in level " << level << ": " << point_indices.back().size() << std::endl;
         std::cout << "Number of domains in level " << level << ": " << fine_grids.back().size() << std::endl;

         ratio = level == n_fine_levels - 1
            ? double(n_coarsest_points) / double(point_indices.back().size())
            : coarse_ratio;
         upward_evaluator.push_back(interpolation::rbf_evaluator<Order>(rbf, -1, common::make_view(points, point_indices.back())));
         point_indices.push_back(divider->choose_coarse_points(ratio));
         upward_evaluator.back().set_field_points(common::make_view(points, point_indices.back()));
      }

      std::cout << "Number of points in coarse: " << point_indices.back().size() << std::endl;
      coarse = std::make_unique<CoarseGrid>(rbf, poly_degree, points, point_indices.back());

      for (int level = 1; level < n_fine_levels; level++) {
         downward_evaluator.push_back(interpolation::rbf_evaluator<Order>(rbf, poly_degree, common::make_view(points, point_indices.back())));
         downward_evaluator.back().set_field_points(common::make_view(points, point_indices[level - 1]));
      }

      if (poly_degree >= 0) {
         polynomial::orthonormal_basis<> poly(poly_degree, points);
         p = poly.evaluate_points(points).transpose();
         ap = Eigen::MatrixXd(p.rows(), p.cols());

         auto finest_evaluator = interpolation::rbf_symmetric_evaluator<Order>(rbf, -1, points);
         for (size_t i = 0; i < p.cols(); i++) {
            finest_evaluator.set_weights(p.col(i));
            ap.col(i) = finest_evaluator.evaluate();
         }
      }
   }

   Eigen::VectorXd operator()(const Eigen::VectorXd& v) const override
   {
      assert(v.size() == size());

      if (n_fine_levels == 0)
         return v;

      Eigen::VectorXd weights_total = Eigen::VectorXd::Zero(size());
      Eigen::VectorXd residuals = v.head(n_points);
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
               const auto& finer_indices = point_indices[level - 1];
               Eigen::VectorXd finer_weights(finer_indices.size());
               for (size_t i = 0; i < finer_indices.size(); i++) {
                  finer_weights(i) = weights(finer_indices[i]);
               }
               upward_evaluator[level].set_weights(finer_weights);
            } else {
               upward_evaluator[level].set_weights(weights);
            }
            auto fit = upward_evaluator[level].evaluate();

            const auto& indices = point_indices[level];
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
               const auto& coarse_indices = point_indices.back();
               Eigen::VectorXd coarse_weights(coarse_indices.size() + n_polynomials);
               for (size_t i = 0; i < coarse_indices.size(); i++) {
                  coarse_weights(i) = weights(coarse_indices[i]);
               }
               coarse_weights.tail(n_polynomials) = weights.tail(n_polynomials);
               downward_evaluator[level].set_weights(coarse_weights);

               auto fit = downward_evaluator[level].evaluate();

               const auto& indices = point_indices[level];
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

   size_t size() const override
   {
      return n_points + n_polynomials;
   }
};

} // namespace preconditioner
} // namespace polatory
