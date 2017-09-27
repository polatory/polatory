// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>

#include <Eigen/Core>

#include "../fmm/fmm_operator.hpp"
#include "../fmm/tree_height.hpp"
#include "../geometry/bbox3.hpp"
#include "../krylov/linear_operator.hpp"
#include "../polynomial/basis_base.hpp"
#include "../polynomial/monomial_basis.hpp"
#include "polynomial_matrix.hpp"
#include "../rbf/rbf_base.hpp"

namespace polatory {
namespace interpolation {

template <int Order = 10>
struct rbf_operator : krylov::linear_operator {
private:
  typedef polynomial_matrix<polynomial::monomial_basis<>> poly_mat;

  const rbf::rbf_base& rbf;
  const int poly_degree;
  const size_t n_polynomials;

  size_t n_points;
  std::unique_ptr<fmm::fmm_operator<Order>> a;
  std::unique_ptr<poly_mat> p;

public:
  template <typename Container>
  rbf_operator(const rbf::rbf_base& rbf, int poly_degree,
               const Container& points)
    : rbf(rbf)
    , poly_degree(poly_degree)
    , n_polynomials(polynomial::basis_base::dimension(poly_degree)) {
    auto bbox = geometry::bbox3d::from_points(points);

    a = std::make_unique<fmm::fmm_operator<Order>>(rbf, fmm::tree_height(points.size()), bbox);

    if (poly_degree >= 0) {
      p = std::make_unique<poly_mat>(poly_degree);
    }

    set_points(points);
  }

  rbf_operator(const rbf::rbf_base& rbf, int poly_degree,
               int tree_height, const geometry::bbox3d& bbox)
    : rbf(rbf)
    , poly_degree(poly_degree)
    , n_polynomials(polynomial::basis_base::dimension(poly_degree))
    , n_points(0) {
    a = std::make_unique<fmm::fmm_operator<Order>>(rbf, tree_height, bbox);

    if (poly_degree >= 0) {
      p = std::make_unique<poly_mat>(poly_degree);
    }
  }

  Eigen::VectorXd operator()(const Eigen::VectorXd& weights) const override {
    assert(weights.size() == size());

    Eigen::VectorXd y = Eigen::VectorXd::Zero(size());

    auto diagonal = rbf.evaluate(0.0) - rbf.nugget();
    y.head(n_points) = diagonal * weights.head(n_points);

    a->set_weights(weights.head(n_points));
    y.head(n_points) += a->evaluate();

    if (poly_degree >= 0) {
      // Add polynomial terms.
      y += p->evaluate(weights);
    }

    return y;
  }

  template <typename Container>
  void set_points(const Container& points) {
    n_points = points.size();

    a->set_points(points);

    if (poly_degree >= 0) {
      p->set_points(points);
    }
  }

  size_t size() const override {
    return n_points + n_polynomials;
  }
};

} // namespace interpolation
} // namespace polatory
