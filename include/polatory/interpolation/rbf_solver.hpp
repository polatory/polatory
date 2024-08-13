#pragma once

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_operator.hpp>
#include <polatory/interpolation/rbf_residual_evaluator.hpp>
#include <polatory/krylov/fgmres.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/preconditioner/ras_preconditioner.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::interpolation {

template <int Dim>
class Solver {
  static constexpr int kDim = Dim;
  using Bbox = geometry::Bbox<kDim>;
  using Model = Model<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Operator = Operator<kDim>;
  using Points = geometry::Points<kDim>;
  using Preconditioner = preconditioner::RasPreconditioner<kDim>;
  using ResidualEvaluator = ResidualEvaluator<kDim>;

 public:
  Solver(const Model& model, const Points& points, const Points& grad_points, double accuracy,
         double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        op_(model, points, grad_points),
        res_eval_(model, points, grad_points, accuracy, grad_accuracy) {
    set_points(points, grad_points);
  }

  Solver(const Model& model, const Bbox& bbox, double accuracy, double grad_accuracy)
      : model_(model),
        l_(model.poly_basis_size()),
        op_(model, bbox),
        res_eval_(model, bbox, accuracy, grad_accuracy) {}

  void set_points(const Points& points) { set_points(points, Points(0, kDim)); }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    op_.set_points(points, grad_points);
    res_eval_.set_points(points, grad_points);

    pc_ = std::make_unique<Preconditioner>(model_, points, grad_points);

    if (l_ > 0) {
      MonomialBasis poly(model_.poly_degree());
      p_ = poly.evaluate(points, grad_points);
      common::orthonormalize_cols(p_);
    }
  }

  template <class DerivedValues, class DerivedInitialWeights = VecX>
  VecX solve(const Eigen::MatrixBase<DerivedValues>& values, double tolerance, int max_iter,
             const Eigen::MatrixBase<DerivedInitialWeights>* initial_weights = nullptr) const {
    return solve(values, tolerance, tolerance, max_iter, initial_weights);
  }

  template <class DerivedValues, class DerivedInitialWeights = VecX>
  VecX solve(const Eigen::MatrixBase<DerivedValues>& values, double tolerance,
             double grad_tolerance, int max_iter,
             const Eigen::MatrixBase<DerivedInitialWeights>* initial_weights = nullptr) const {
    POLATORY_ASSERT(values.rows() == mu_ + kDim * sigma_);
    POLATORY_ASSERT(initial_weights == nullptr ||
                    initial_weights->rows() == mu_ + kDim * sigma_ + l_);

    VecX weights = VecX::Zero(mu_ + kDim * sigma_ + l_);

    if (initial_weights != nullptr) {
      weights = *initial_weights;

      if (l_ > 0) {
        // Orthogonalize weights against P.
        VecX dot = p_.transpose() * weights.head(mu_ + kDim * sigma_);
        weights.head(mu_ + kDim * sigma_) -= p_ * dot;
      }
    }

    VecX rhs(mu_ + kDim * sigma_ + l_);
    rhs.head(mu_ + kDim * sigma_) = values;
    rhs.tail(l_) = VecX::Zero(l_);

    krylov::Fgmres solver(op_, rhs, max_iter);
    solver.set_initial_solution(weights);
    solver.set_right_preconditioner(*pc_);
    solver.setup();

    // The solver does not work if the initial solution is already the solution.
    if (solver.relative_residual() == 0.0) {
      return weights;
    }

    res_eval_.set_values(values);

    std::cout << std::setw(8) << "iter"            //
              << std::setw(16) << "residual"       //
              << std::setw(16) << "grad_residual"  //
              << std::endl;

    while (true) {
      weights = solver.solution_vector();

      auto convergence = res_eval_.converged(weights, tolerance, grad_tolerance);

      auto prefix = convergence.exact_residual ? "" : "~";
      auto grad_prefix = convergence.exact_grad_residual ? "" : "~";
      std::cout << std::scientific                                                            //
                << std::setw(8) << solver.iteration_count()                                   //
                << std::setw(4) << prefix << std::setw(12) << convergence.residual            //
                << std::setw(4) << grad_prefix << std::setw(12) << convergence.grad_residual  //
                << std::endl                                                                  //
                << std::defaultfloat;

      if (convergence.converged) {
        break;
      }

      if (solver.iteration_count() == solver.max_iterations()) {
        throw std::runtime_error("reached the maximum number of iterations");
      }

      solver.iterate_process();
    }

    return weights;
  }

 private:
  const Model& model_;
  const Index l_;

  Index mu_{};
  Index sigma_{};
  Operator op_;
  mutable ResidualEvaluator res_eval_;
  std::unique_ptr<Preconditioner> pc_;
  MatX p_;
};

}  // namespace polatory::interpolation
