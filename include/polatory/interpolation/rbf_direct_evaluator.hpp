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

class rbf_direct_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis<>>;

public:
  rbf_direct_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                       const std::vector<Eigen::Vector3d>& source_points)
    : rbf_(rbf)
    , n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , src_points_(source_points) {
    if (n_polynomials_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
    }
  }

  Eigen::VectorXd evaluate() const {
    auto y_accum = std::vector<numeric::kahan_sum_accumulator<double>>(fld_points_.size());

    for (size_t i = 0; i < src_points_.size(); i++) {
      for (size_t j = 0; j < fld_points_.size(); j++) {
        auto a_ij = rbf_.evaluate(src_points_[i], fld_points_[j]);
        y_accum[j] += weights_(i) * a_ij;
      }
    }

    if (n_polynomials_ > 0) {
      // Add polynomial terms.
      auto poly_val = p_->evaluate();
      for (size_t i = 0; i < fld_points_.size(); i++) {
        y_accum[i] += poly_val(i);
      }
    }

    Eigen::VectorXd y(fld_points_.size());
    for (size_t i = 0; i < fld_points_.size(); i++) {
      y(i) = y_accum[i].get();
    }

    return y;
  }

  void set_field_points(const std::vector<Eigen::Vector3d>& field_points) {
    this->fld_points_ = field_points;

    if (n_polynomials_ > 0) {
      p_->set_field_points(fld_points_);
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.size() == src_points_.size() + n_polynomials_);

    this->weights_ = weights;

    if (n_polynomials_ > 0) {
      p_->set_weights(weights.tail(n_polynomials_));
    }
  }

private:
  const rbf::rbf_base& rbf_;
  const size_t n_polynomials_;

  std::unique_ptr<PolynomialEvaluator> p_;

  const std::vector<Eigen::Vector3d> src_points_;
  std::vector<Eigen::Vector3d> fld_points_;
  Eigen::VectorXd weights_;
};

} // namespace interpolation
} // namespace polatory
