#pragma once

#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/common/orthonormalize.hpp>
#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_operator.hpp>
#include <polatory/interpolation/rbf_residual_evaluator.hpp>
#include <polatory/krylov/fgmres.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/precision.hpp>
#include <polatory/preconditioner/ras_preconditioner.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Dim>
class rbf_solver {
  static constexpr int kDim = Dim;
  using Bbox = geometry::bboxNd<kDim>;
  using Model = model<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using Operator = rbf_operator<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Preconditioner = preconditioner::ras_preconditioner<kDim>;
  using ResidualEvaluator = rbf_residual_evaluator<kDim>;

 public:
  rbf_solver(const Model& model, const Points& points)
      : rbf_solver(model, points, Points(0, kDim)) {}

  rbf_solver(const Model& model, const Points& points, const Points& grad_points)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        op_(model, points, grad_points, precision::kPrecise),
        res_eval_(model, points, grad_points) {
    set_points(points, grad_points);
  }

  rbf_solver(const Model& model, const Bbox& bbox)
      : model_(model),
        l_(model.poly_basis_size()),
        op_(model, bbox, precision::kPrecise),
        res_eval_(model, bbox) {}

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

  template <class DerivedValues, class DerivedInitialWeights = common::valuesd>
  common::valuesd solve(
      const Eigen::MatrixBase<DerivedValues>& values, double absolute_tolerance, int max_iter,
      const Eigen::MatrixBase<DerivedInitialWeights>* initial_weights = nullptr) const {
    return solve(values, absolute_tolerance, absolute_tolerance, max_iter, initial_weights);
  }

  template <class DerivedValues, class DerivedInitialWeights = common::valuesd>
  common::valuesd solve(
      const Eigen::MatrixBase<DerivedValues>& values, double absolute_tolerance,
      double grad_absolute_tolerance, int max_iter,
      const Eigen::MatrixBase<DerivedInitialWeights>* initial_weights = nullptr) const {
    POLATORY_ASSERT(values.rows() == mu_ + kDim * sigma_);
    POLATORY_ASSERT(initial_weights == nullptr ||
                    initial_weights->rows() == mu_ + kDim * sigma_ + l_);

    common::valuesd weights = common::valuesd::Zero(mu_ + kDim * sigma_ + l_);

    // The solver does not work when all values are zero.
    if (values.isZero()) {
      return weights;
    }

    if (initial_weights != nullptr) {
      weights = *initial_weights;

      if (l_ > 0) {
        // Orthogonalize weights against P.
        common::valuesd dot = p_.transpose() * weights.head(mu_ + kDim * sigma_);
        weights.head(mu_ + kDim * sigma_) -= p_ * dot;
      }
    }

    common::valuesd rhs(mu_ + kDim * sigma_ + l_);
    rhs.head(mu_ + kDim * sigma_) = values;
    rhs.tail(l_) = common::valuesd::Zero(l_);

    krylov::fgmres solver(op_, rhs, max_iter);
    solver.set_initial_solution(weights);
    solver.set_right_preconditioner(*pc_);
    solver.setup();

    std::cout << std::setw(4) << "iter" << std::setw(16) << "rel_res" << std::endl
              << std::setw(4) << solver.iteration_count() << std::setw(16) << std::scientific
              << solver.relative_residual() << std::defaultfloat << std::endl;

    while (true) {
      solver.iterate_process();
      weights = solver.solution_vector();
      std::cout << std::setw(4) << solver.iteration_count() << std::setw(16) << std::scientific
                << solver.relative_residual() << std::defaultfloat << std::endl;

      auto [converged, res, grad_res] =
          res_eval_.converged(values, weights, absolute_tolerance, grad_absolute_tolerance);
      if (converged) {
        if (mu_ > 0) {
          std::cout << "Achieved absolute residual: " << res << std::endl;
        }
        if (sigma_ > 0) {
          std::cout << "Achieved absolute grad residual: " << grad_res << std::endl;
        }
        break;
      }

      if (solver.iteration_count() == solver.max_iterations()) {
        std::cerr
            << "Warning: reached the maximum number of iterations, returning the current solution."
            << std::endl;
        break;
      }
    }

    return weights;
  }

 private:
  const Model& model_;
  const index_t l_;

  index_t mu_{};
  index_t sigma_{};
  Operator op_;
  ResidualEvaluator res_eval_;
  std::unique_ptr<Preconditioner> pc_;
  matrixd p_;
};

}  // namespace polatory::interpolation
