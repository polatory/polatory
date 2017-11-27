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

class rbf_direct_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis<>>;

public:
  rbf_direct_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                       const geometry::points3d& source_points);

  Eigen::VectorXd evaluate() const;

  size_t field_size() const;

  void set_field_points(const geometry::points3d& field_points);

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.size() == source_size() + n_polynomials_);

    this->weights_ = weights;

    if (n_polynomials_ > 0) {
      p_->set_weights(weights.tail(n_polynomials_));
    }
  }

  size_t source_size() const;

private:
  const rbf::rbf_base& rbf_;
  const size_t n_polynomials_;
  const geometry::points3d src_points_;

  std::unique_ptr<PolynomialEvaluator> p_;

  geometry::points3d fld_points_;
  Eigen::VectorXd weights_;
};

} // namespace interpolation
} // namespace polatory
