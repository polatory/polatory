// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <iomanip>
#include <iostream>
#include <memory>

#include <Eigen/Core>

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_operator.hpp>
#include <polatory/interpolation/rbf_residual_evaluator.hpp>
#include <polatory/krylov/fgmres.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/orthonormal_basis.hpp>
#include <polatory/preconditioner/ras_preconditioner.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace interpolation {

class rbf_solver {
  using Preconditioner = preconditioner::ras_preconditioner;

public:
  rbf_solver(const model& model, const geometry::points3d& points)
    : model_(model)
    , n_poly_basis_(model.poly_basis_size())
    , n_points_(static_cast<index_t>(points.rows())) {
    op_ = std::make_unique<rbf_operator<>>(model, points);
    res_eval_ = std::make_unique<rbf_residual_evaluator>(model, points);

    set_points(points);
  }

  rbf_solver(const model& model, int tree_height, const geometry::bbox3d& bbox)
    : model_(model)
    , n_poly_basis_(model.poly_basis_size())
    , n_points_(0) {
    op_ = std::make_unique<rbf_operator<>>(model, tree_height, bbox);
    res_eval_ = std::make_unique<rbf_residual_evaluator>(model, tree_height, bbox);
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = static_cast<index_t>(points.rows());

    op_->set_points(points);
    res_eval_->set_points(points);

    pc_ = std::make_unique<Preconditioner>(model_, points);

    if (n_poly_basis_ > 0) {
      polynomial::orthonormal_basis poly(model_.poly_dimension(), model_.poly_degree(), points);
      p_ = poly.evaluate_points(points).transpose();
    }
  }

  template <class Derived>
  common::valuesd solve(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance) const {
    assert(values.rows() == n_points_);

    return solve_impl(values, absolute_tolerance);
  }

  template <class Derived, class Derived2>
  common::valuesd solve(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance,
                        const Eigen::MatrixBase<Derived2>& initial_solution) const {
    assert(values.rows() == n_points_);
    assert(initial_solution.rows() == n_points_ + n_poly_basis_);

    common::valuesd ini_sol = initial_solution;

    if (n_poly_basis_ > 0) {
      // Orthogonalize weights against P.
      auto n_cols = static_cast<index_t>(p_.cols());
      for (index_t i = 0; i < n_cols; i++) {
        ini_sol.head(n_points_) -= p_.col(i).dot(ini_sol.head(n_points_)) * p_.col(i);
      }
    }

    return solve_impl(values, absolute_tolerance, &ini_sol);
  }

private:
  template <class Derived, class Derived2 = common::valuesd>
  common::valuesd solve_impl(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance,
                             const Eigen::MatrixBase<Derived2> *initial_solution = nullptr) const {
    common::valuesd rhs(n_points_ + n_poly_basis_);
    rhs.head(n_points_) = values;
    rhs.tail(n_poly_basis_) = common::valuesd::Zero(n_poly_basis_);

    krylov::fgmres solver(*op_, rhs, 32);
    if (initial_solution != nullptr)
      solver.set_initial_solution(*initial_solution);
    solver.set_right_preconditioner(*pc_);
    solver.setup();

    std::cout << std::setw(4) << "iter"
              << std::setw(16) << "rel_res" << std::endl
              << std::setw(4) << solver.iteration_count()
              << std::setw(16) << std::scientific << solver.relative_residual() << std::defaultfloat << std::endl;

    common::valuesd solution;
    while (true) {
      solver.iterate_process();
      solution = solver.solution_vector();
      std::cout << std::setw(4) << solver.iteration_count()
                << std::setw(16) << std::scientific << solver.relative_residual() << std::defaultfloat << std::endl;

      auto convergence = res_eval_->converged(values, solution, absolute_tolerance);
      if (convergence.first) {
        std::cout << "Achieved absolute residual: " << convergence.second << std::endl;
        break;
      }

      if (solver.iteration_count() == solver.max_iterations()) {
        std::cout << "Reached the maximum number of iterations." << std::endl;
        break;
      }
    }

    return solution;
  }

  const model model_;
  const index_t n_poly_basis_;

  index_t n_points_;
  std::unique_ptr<rbf_operator<>> op_;
  std::unique_ptr<Preconditioner> pc_;
  std::unique_ptr<rbf_residual_evaluator> res_eval_;
  Eigen::MatrixXd p_;
};

}  // namespace interpolation
}  // namespace polatory
