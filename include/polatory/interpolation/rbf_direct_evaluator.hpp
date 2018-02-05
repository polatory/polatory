// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>

namespace polatory {
namespace interpolation {

class rbf_direct_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

public:
  rbf_direct_evaluator(const model& model, const geometry::points3d& source_points);

  common::valuesd evaluate() const;

  void set_field_points(const geometry::points3d& field_points);

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.rows() == n_src_points_ + n_poly_basis_);

    weights_ = weights;

    if (n_poly_basis_ > 0) {
      p_->set_weights(weights.tail(n_poly_basis_));
    }
  }

private:
  const model model_;
  const size_t n_poly_basis_;
  const size_t n_src_points_;
  const geometry::points3d src_points_;

  std::unique_ptr<PolynomialEvaluator> p_;

  size_t n_fld_points_;
  geometry::points3d fld_points_;
  common::valuesd weights_;
};

}  // namespace interpolation
}  // namespace polatory
