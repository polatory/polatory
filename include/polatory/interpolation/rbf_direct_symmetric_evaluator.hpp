// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace interpolation {

class rbf_direct_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis<>>;

public:
  rbf_direct_symmetric_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                                 const geometry::points3d& points);

  Eigen::VectorXd evaluate() const;

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

  const geometry::points3d points_;
  Eigen::VectorXd weights_;
};

} // namespace interpolation
} // namespace polatory
