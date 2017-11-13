// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>

#include <Eigen/Core>

#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/fmm/tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {
namespace interpolation {

template <int Order = 10>
class rbf_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis<>>;

public:
  template <class Container>
  rbf_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                const Container& source_points)
    : n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree)) {
    auto bbox = geometry::bbox3d::from_points(source_points);

    a_ = std::make_unique<fmm::fmm_evaluator<Order>>(rbf, fmm::tree_height(source_points.size()), bbox);

    if (n_polynomials_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
    }

    set_source_points(source_points);
  }

  template <class Container>
  rbf_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                const Container& source_points, const geometry::bbox3d& bbox)
    : n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree)) {
    a_ = std::make_unique<fmm::fmm_evaluator<Order>>(rbf, fmm::tree_height(source_points.size()), bbox);

    if (n_polynomials_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
    }

    set_source_points(source_points);
  }

  rbf_evaluator(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree,
                int tree_height, const geometry::bbox3d& bbox)
    : n_polynomials_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , n_src_points_(0) {
    a_ = std::make_unique<fmm::fmm_evaluator<Order>>(rbf, tree_height, bbox);

    if (n_polynomials_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
    }
  }

  Eigen::VectorXd evaluate() const {
    auto y = a_->evaluate();

    if (n_polynomials_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  template <class Container>
  Eigen::VectorXd evaluate_points(const Container& field_points) const {
    set_field_points(field_points);
    return evaluate();
  }

  template <class Container>
  void set_field_points(const Container& points) const {
    a_->set_field_points(points);

    if (n_polynomials_ > 0) {
      p_->set_field_points(points);
    }
  }

  template <class Container>
  void set_source_points(const Container& points) {
    n_src_points_ = points.size();

    a_->set_source_points(points);
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) const {
    assert(weights.size() == n_src_points_ + n_polynomials_);

    a_->set_weights(weights.head(n_src_points_));

    if (n_polynomials_ > 0) {
      p_->set_weights(weights.tail(n_polynomials_));
    }
  }

private:
  const size_t n_polynomials_;

  size_t n_src_points_;
  std::unique_ptr<fmm::fmm_evaluator<Order>> a_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

} // namespace interpolation
} // namespace polatory
