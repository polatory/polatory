// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <vector>

#include "polatory/numeric/sum_accumulator.hpp"
#include "polatory/polynomial/basis_base.hpp"
#include "polatory/polynomial/monomial_basis.hpp"
#include "polatory/polynomial/polynomial_evaluator.hpp"
#include "polatory/rbf/rbf_base.hpp"

namespace polatory {
namespace interpolation {

class rbf_direct_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis<>>;

public:
  rbf_direct_symmetric_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                                 const std::vector<Eigen::Vector3d>& points)
    : rbf_(rbf)
    , n_points_(points.size())
    , n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , points_(points) {
    if (n_polynomials_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
      p_->set_field_points(points);
    }
  }

  Eigen::VectorXd evaluate() const {
    auto y_accum = std::vector<numeric::kahan_sum_accumulator<double>>(n_points_);

    auto rbf_at_center = rbf_.evaluate(0.0);
    for (size_t i = 0; i < n_points_; i++) {
      y_accum[i] += weights_(i) * rbf_at_center;
    }
    for (size_t i = 0; i < n_points_ - 1; i++) {
      for (size_t j = i + 1; j < n_points_; j++) {
        auto a_ij = rbf_.evaluate(points_[i], points_[j]);
        y_accum[i] += weights_(j) * a_ij;
        y_accum[j] += weights_(i) * a_ij;
      }
    }

    if (n_polynomials_ > 0) {
      // Add polynomial terms.
      auto poly_val = p_->evaluate();
      for (size_t i = 0; i < n_points_; i++) {
        y_accum[i] += poly_val(i);
      }
    }

    Eigen::VectorXd y(n_points_);
    for (size_t i = 0; i < n_points_; i++) {
      y(i) = y_accum[i].get();
    }

    return y;
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.size() == n_points_ + n_polynomials_);

    this->weights_ = weights;

    if (n_polynomials_ > 0) {
      p_->set_weights(weights.tail(n_polynomials_));
    }
  }

private:
  const rbf::rbf_base& rbf_;
  const size_t n_points_;
  const size_t n_polynomials_;

  std::unique_ptr<PolynomialEvaluator> p_;

  const std::vector<Eigen::Vector3d> points_;
  Eigen::VectorXd weights_;
};

} // namespace interpolation
} // namespace polatory
