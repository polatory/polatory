// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>

#include <Eigen/Core>

#include <polatory/common/types.hpp>
#include <polatory/fmm/fmm_operator.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>

namespace polatory {
namespace interpolation {

template <int Order = 10>
class rbf_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

public:
  rbf_symmetric_evaluator(const model& model, const geometry::points3d& points)
    : model_(model)
    , n_points_(points.rows())
    , n_poly_basis_(model.poly_basis_size()) {
    auto bbox = geometry::bbox3d::from_points(points);

    a_ = std::make_unique<fmm::fmm_operator<Order>>(model, fmm::fmm_tree_height(points.rows()), bbox);
    a_->set_points(points);

    if (n_poly_basis_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
      p_->set_field_points(points);
    }
  }

  common::valuesd evaluate() const {
    auto rbf_at_zero = model_.rbf().evaluate(0.0);
    common::valuesd y = weights_.head(n_points_) * rbf_at_zero;

    y += a_->evaluate();

    if (n_poly_basis_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.rows() == n_points_ + n_poly_basis_);

    weights_ = weights;

    a_->set_weights(weights.head(n_points_));

    if (n_poly_basis_ > 0) {
      p_->set_weights(weights.tail(n_poly_basis_));
    }
  }

private:
  const model model_;
  const size_t n_points_;
  const size_t n_poly_basis_;

  std::unique_ptr<fmm::fmm_operator<Order>> a_;
  std::unique_ptr<PolynomialEvaluator> p_;

  common::valuesd weights_;
};

}  // namespace interpolation
}  // namespace polatory
