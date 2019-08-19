// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <Eigen/Core>

#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace interpolation {

class rbf_direct_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

public:
  rbf_direct_symmetric_evaluator(const model& model, const geometry::points3d& points);

  common::valuesd evaluate() const;

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(static_cast<index_t>(weights.rows()) == n_points_ + n_poly_basis_);

    weights_ = weights;

    if (n_poly_basis_ > 0) {
      p_->set_weights(weights.tail(n_poly_basis_));
    }
  }

private:
  const model model_;
  const index_t n_points_;
  const index_t n_poly_basis_;

  std::unique_ptr<PolynomialEvaluator> p_;

  const geometry::points3d points_;
  common::valuesd weights_;
};

}  // namespace interpolation
}  // namespace polatory
