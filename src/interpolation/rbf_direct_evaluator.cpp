// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/interpolation/rbf_direct_evaluator.hpp>

#include <polatory/numeric/sum_accumulator.hpp>

namespace polatory {
namespace interpolation {

rbf_direct_evaluator::rbf_direct_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                                           const geometry::points3d& source_points)
  : rbf_(rbf)
  , n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
  , src_points_(source_points) {
  if (n_polynomials_ > 0) {
    p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
  }
}

Eigen::VectorXd rbf_direct_evaluator::evaluate() const {
  auto y_accum = std::vector<numeric::kahan_sum_accumulator<double>>(fld_points_.rows());

  for (size_t i = 0; i < source_size(); i++) {
    for (size_t j = 0; j < field_size(); j++) {
      auto a_ij = rbf_.evaluate(src_points_.row(i), fld_points_.row(j));
      y_accum[j] += weights_(i) * a_ij;
    }
  }

  if (n_polynomials_ > 0) {
    // Add polynomial terms.
    auto poly_val = p_->evaluate();
    for (size_t i = 0; i < field_size(); i++) {
      y_accum[i] += poly_val(i);
    }
  }

  Eigen::VectorXd y(field_size());
  for (size_t i = 0; i < field_size(); i++) {
    y(i) = y_accum[i].get();
  }

  return y;
}

size_t rbf_direct_evaluator::field_size() const {
  return fld_points_.rows();
}

void rbf_direct_evaluator::set_field_points(const geometry::points3d& field_points) {
  fld_points_ = field_points;

  if (n_polynomials_ > 0) {
    p_->set_field_points(fld_points_);
  }
}

size_t rbf_direct_evaluator::source_size() const {
  return src_points_.rows();
}

} // namespace interpolation
} // namespace polatory
