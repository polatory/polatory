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
  typedef polynomial::polynomial_evaluator<polynomial::monomial_basis<>> poly_eval;

  const rbf::rbf_base& rbf;
  const int poly_degree;
  const size_t n_polynomials;

  std::unique_ptr<poly_eval> p;

  const std::vector<Eigen::Vector3d> src_points;
  std::vector<Eigen::Vector3d> fld_points;
  Eigen::VectorXd weights;

public:
  rbf_direct_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                       const std::vector<Eigen::Vector3d>& source_points)
    : rbf(rbf)
    , poly_degree(poly_degree)
    , n_polynomials(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , src_points(source_points) {
    if (poly_degree >= 0) {
      p = std::make_unique<poly_eval>(poly_dimension, poly_degree);
    }
  }

  Eigen::VectorXd evaluate() const {
    auto y_accum = std::vector<numeric::kahan_sum_accumulator<double>>(fld_points.size());

    for (size_t i = 0; i < src_points.size(); i++) {
      for (size_t j = 0; j < fld_points.size(); j++) {
        auto a_ij = rbf.evaluate(src_points[i], fld_points[j]);
        y_accum[j] += weights(i) * a_ij;
      }
    }

    if (poly_degree >= 0) {
      // Add polynomial terms.
      auto poly_val = p->evaluate();
      for (size_t i = 0; i < fld_points.size(); i++) {
        y_accum[i] += poly_val(i);
      }
    }

    Eigen::VectorXd y(fld_points.size());
    for (size_t i = 0; i < fld_points.size(); i++) {
      y(i) = y_accum[i].get();
    }

    return y;
  }

  void set_field_points(const std::vector<Eigen::Vector3d>& field_points) {
    this->fld_points = field_points;

    if (poly_degree >= 0) {
      p->set_field_points(fld_points);
    }
  }

  template <typename Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.size() == src_points.size() + n_polynomials);

    this->weights = weights;

    if (poly_degree >= 0) {
      p->set_weights(weights.tail(n_polynomials));
    }
  }
};

} // namespace interpolation
} // namespace polatory
