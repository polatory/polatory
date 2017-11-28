// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/interpolation/rbf_direct_symmetric_evaluator.hpp>

#include <polatory/numeric/sum_accumulator.hpp>

namespace polatory {
namespace interpolation {

rbf_direct_symmetric_evaluator::rbf_direct_symmetric_evaluator(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
                                                               const geometry::points3d& points)
  : rbf_(rbf)
  , n_points_(points.rows())
  , n_poly_basis_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
  , points_(points) {
  if (n_poly_basis_ > 0) {
    p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
    p_->set_field_points(points);
  }
}

common::valuesd rbf_direct_symmetric_evaluator::evaluate() const {
  auto y_accum = std::vector<numeric::kahan_sum_accumulator<double>>(n_points_);

  auto& rbf_kern = rbf_.get();
  auto rbf_at_zero = rbf_kern.evaluate(0.0);
  for (size_t i = 0; i < n_points_; i++) {
    y_accum[i] += weights_(i) * rbf_at_zero;
  }
  for (size_t i = 0; i < n_points_ - 1; i++) {
    for (size_t j = i + 1; j < n_points_; j++) {
      auto a_ij = rbf_kern.evaluate(points_.row(i), points_.row(j));
      y_accum[i] += weights_(j) * a_ij;
      y_accum[j] += weights_(i) * a_ij;
    }
  }

  if (n_poly_basis_ > 0) {
    // Add polynomial terms.
    auto poly_val = p_->evaluate();
    for (size_t i = 0; i < n_points_; i++) {
      y_accum[i] += poly_val(i);
    }
  }

  common::valuesd y(n_points_);
  for (size_t i = 0; i < n_points_; i++) {
    y(i) = y_accum[i].get();
  }

  return y;
}

} // namespace interpolation
} // namespace polatory
