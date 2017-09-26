// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <Eigen/Core>

#include "../geometry/bbox3.hpp"
#include "../krylov.hpp"
#include "../polynomial.hpp"
#include "../preconditioner/ras_preconditioner.hpp"
#include "../rbf/rbf_base.hpp"
#include "rbf_residual_evaluator.hpp"
#include "rbf_operator.hpp"

namespace polatory {
namespace interpolation {

class rbf_solver {
   const rbf::rbf_base& rbf;
   const int poly_degree;
   const size_t n_polynomials;

   size_t n_points;
   std::unique_ptr<rbf_operator<>> op;
   std::unique_ptr<preconditioner::ras_preconditioner> pc;
   std::unique_ptr<rbf_residual_evaluator> res_eval;
   Eigen::MatrixXd p;

   template<typename Derived, typename Derived2 = Eigen::VectorXd>
   Eigen::VectorXd solve_impl(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance, const Eigen::MatrixBase<Derived2> *initial_solution = nullptr) const
   {
      Eigen::VectorXd rhs(n_points + n_polynomials);
      rhs.head(n_points) = values;
      rhs.tail(n_polynomials) = Eigen::VectorXd::Zero(n_polynomials);

      krylov::fgmres solver(*op, rhs, 32);
      if (initial_solution != nullptr)
         solver.set_initial_solution(*initial_solution);
      solver.set_right_preconditioner(*pc);
      solver.setup();

      Eigen::VectorXd solution;
      while (true) {
         std::cout << solver.iteration_count() << ": \t" << solver.relative_residual() << std::endl;
         solver.iterate_process();
         solution = solver.solution_vector();
         if (res_eval->converged(values, solution, absolute_tolerance) ||
            solver.iteration_count() == solver.max_iterations())
            break;
      }
      std::cout << solver.iteration_count() << ": \t" << solver.relative_residual() << std::endl;

      return solution;
   }

public:
   template<typename Container>
   rbf_solver(const rbf::rbf_base& rbf, int poly_degree,
      const Container& points)
      : rbf(rbf)
      , poly_degree(poly_degree)
      , n_polynomials(polynomial::basis_base::dimension(poly_degree))
      , n_points(points.size())
   {
      op = std::make_unique<rbf_operator<>>(rbf, poly_degree, points);
      pc = std::make_unique<preconditioner::ras_preconditioner>(rbf, poly_degree, points);
      res_eval = std::make_unique<rbf_residual_evaluator>(rbf, poly_degree, points);

      if (poly_degree >= 0) {
         polynomial::orthonormal_basis<> poly(poly_degree, points);
         p = poly.evaluate_points(points).transpose();
      }
   }

   rbf_solver(const rbf::rbf_base& rbf, int poly_degree,
      int tree_height, const geometry::bbox3d& bbox)
      : rbf(rbf)
      , poly_degree(poly_degree)
      , n_polynomials(polynomial::basis_base::dimension(poly_degree))
      , n_points(0)
   {
      op = std::make_unique<rbf_operator<>>(rbf, poly_degree, tree_height, bbox);
      res_eval = std::make_unique<rbf_residual_evaluator>(rbf, poly_degree, tree_height, bbox);
   }

   template<typename Container>
   void set_points(const Container& points)
   {
      n_points = points.size();

      op->set_points(points);
      res_eval->set_points(points);

      pc = std::make_unique<preconditioner::ras_preconditioner>(rbf, poly_degree, points);

      if (poly_degree >= 0) {
         polynomial::orthonormal_basis<> poly(poly_degree, points);
         p = poly.evaluate_points(points).transpose();
      }
   }

   template<typename Derived>
   Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance) const
   {
      assert(values.size() == n_points);

      return solve_impl(values, absolute_tolerance);
   }

   template<typename Derived, typename Derived2>
   Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance, const Eigen::MatrixBase<Derived2>& initial_solution) const
   {
      assert(values.size() == n_points);
      assert(initial_solution.size() == n_points + n_polynomials);

      Eigen::VectorXd ini_sol = initial_solution;

      if (poly_degree >= 0) {
         // Orthogonalize weights against P.
         for (size_t i = 0; i < p.cols(); i++) {
            ini_sol.head(n_points) -= p.col(i).dot(ini_sol.head(n_points)) * p.col(i);
         }
      }

      return solve_impl(values, absolute_tolerance, &ini_sol);
   }
};

} // namespace interpolation
} // namespace polatory
