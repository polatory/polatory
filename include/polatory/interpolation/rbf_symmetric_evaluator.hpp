// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include <polatory/fmm/fmm_operator.hpp>
#include <polatory/fmm/tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace interpolation {

template <int Order = 10>
class rbf_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis<>>;

public:
  rbf_symmetric_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                          const std::vector<Eigen::Vector3d>& points)
    : rbf_(rbf)
    , n_points_(points.size())
    , n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree)) {
    auto bbox = geometry::bbox3d::from_points(points);

    a_ = std::make_unique<fmm::fmm_operator<Order>>(rbf, fmm::tree_height(points.size()), bbox);
    a_->set_points(points);

    if (n_polynomials_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
      p_->set_field_points(points);
    }
  }

  Eigen::VectorXd evaluate() const {
    auto rbf_at_center = rbf_.evaluate(0.0);
    Eigen::VectorXd y = weights_.head(n_points_) * rbf_at_center;

    y += a_->evaluate();

    if (n_polynomials_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.size() == n_points_ + n_polynomials_);

    this->weights_ = weights;

    a_->set_weights(weights.head(n_points_));

    if (n_polynomials_ > 0) {
      p_->set_weights(weights.tail(n_polynomials_));
    }
  }

private:
  const rbf::rbf_base& rbf_;
  const size_t n_points_;
  const size_t n_polynomials_;

  std::unique_ptr<fmm::fmm_operator<Order>> a_;
  std::unique_ptr<PolynomialEvaluator> p_;

  Eigen::VectorXd weights_;
};

} // namespace interpolation
} // namespace polatory
