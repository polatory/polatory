// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>

#include <Eigen/Core>

#include "polatory/fmm/fmm_evaluator.hpp"
#include "polatory/fmm/tree_height.hpp"
#include "polatory/geometry/bbox3d.hpp"
#include "polatory/polynomial/basis_base.hpp"
#include "polatory/polynomial/monomial_basis.hpp"
#include "polatory/polynomial/polynomial_evaluator.hpp"
#include "polatory/rbf/rbf_base.hpp"

namespace polatory {
namespace interpolation {

template <int Order = 10>
class rbf_evaluator {
  typedef polynomial::polynomial_evaluator<polynomial::monomial_basis<>> poly_eval;

  const int poly_degree;
  const size_t n_polynomials;

  size_t n_src_points;
  std::unique_ptr<fmm::fmm_evaluator<Order>> a;
  std::unique_ptr<poly_eval> p;

  Eigen::VectorXd weights;

public:
  template <typename Container>
  rbf_evaluator(const rbf::rbf_base& rbf, int poly_degree,
                const Container& source_points)
    : poly_degree(poly_degree)
    , n_polynomials(polynomial::basis_base::dimension(poly_degree)) {
    auto bbox = geometry::bbox3d::from_points(source_points);

    a = std::make_unique<fmm::fmm_evaluator<Order>>(rbf, fmm::tree_height(source_points.size()), bbox);

    if (poly_degree >= 0) {
      p = std::make_unique<poly_eval>(poly_degree);
    }

    set_source_points(source_points);
  }

  template <typename Container>
  rbf_evaluator(const rbf::rbf_base& rbf, int poly_degree,
                const Container& source_points, const geometry::bbox3d& bbox)
    : poly_degree(poly_degree)
    , n_polynomials(polynomial::basis_base::dimension(poly_degree)) {
    a = std::make_unique<fmm::fmm_evaluator<Order>>(rbf, fmm::tree_height(source_points.size()), bbox);

    if (poly_degree >= 0) {
      p = std::make_unique<poly_eval>(poly_degree);
    }

    set_source_points(source_points);
  }

  rbf_evaluator(const rbf::rbf_base& rbf, int poly_degree,
                int tree_height, const geometry::bbox3d& bbox)
    : poly_degree(poly_degree)
    , n_polynomials(polynomial::basis_base::dimension(poly_degree))
    , n_src_points(0) {
    a = std::make_unique<fmm::fmm_evaluator<Order>>(rbf, tree_height, bbox);

    if (poly_degree >= 0) {
      p = std::make_unique<poly_eval>(poly_degree);
    }
  }

  Eigen::VectorXd evaluate() const {
    auto y = a->evaluate();

    if (poly_degree >= 0) {
      // Add polynomial terms.
      y += p->evaluate();
    }

    return y;
  }

  template <typename Container>
  Eigen::VectorXd evaluate_points(const Container& field_points) const {
    set_field_points(field_points);
    return evaluate();
  }

  template <typename Container>
  void set_field_points(const Container& points) const {
    a->set_field_points(points);

    if (poly_degree >= 0) {
      p->set_field_points(points);
    }
  }

  template <typename Container>
  void set_source_points(const Container& points) {
    n_src_points = points.size();

    a->set_source_points(points);
  }

  template <typename Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) const {
    assert(weights.size() == n_src_points + n_polynomials);

    a->set_weights(weights.head(n_src_points));

    if (poly_degree >= 0) {
      p->set_weights(weights.tail(n_polynomials));
    }
  }
};

} // namespace interpolation
} // namespace polatory
