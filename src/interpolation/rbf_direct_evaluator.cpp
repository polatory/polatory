// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/interpolation/rbf_direct_evaluator.hpp>

#include <polatory/numeric/sum_accumulator.hpp>

namespace polatory {
namespace interpolation {

rbf_direct_evaluator::rbf_direct_evaluator(const model& model, const geometry::points3d& source_points)
  : model_(model)
  , n_poly_basis_(model.poly_basis_size())
  , n_src_points_(source_points.rows())
  , src_points_(source_points) {
  if (n_poly_basis_ > 0) {
    p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
  }
}

common::valuesd rbf_direct_evaluator::evaluate() const {
  auto y_accum = std::vector<numeric::kahan_sum_accumulator<double>>(fld_points_.rows());

  auto& rbf = model_.rbf();
  for (size_t i = 0; i < n_src_points_; i++) {
    for (size_t j = 0; j < n_fld_points_; j++) {
      auto a_ij = rbf.evaluate(src_points_.row(i), fld_points_.row(j));
      y_accum[j] += weights_(i) * a_ij;
    }
  }

  if (n_poly_basis_ > 0) {
    // Add polynomial terms.
    auto poly_val = p_->evaluate();
    for (size_t i = 0; i < n_fld_points_; i++) {
      y_accum[i] += poly_val(i);
    }
  }

  common::valuesd y(n_fld_points_);
  for (size_t i = 0; i < n_fld_points_; i++) {
    y(i) = y_accum[i].get();
  }

  return y;
}

void rbf_direct_evaluator::set_field_points(const geometry::points3d& field_points) {
  n_fld_points_ = field_points.rows();
  fld_points_ = field_points;

  if (n_poly_basis_ > 0) {
    p_->set_field_points(fld_points_);
  }
}

}  // namespace interpolation
}  // namespace polatory
