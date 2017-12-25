// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <memory>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/fmm/fmm_operator.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/interpolation/polynomial_matrix.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory {
namespace interpolation {

template <int Order = 10>
struct rbf_operator : krylov::linear_operator {
private:
  using PolynomialEvaluator = polynomial_matrix<polynomial::monomial_basis>;

public:
  rbf_operator(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
               const geometry::points3d& points)
    : rbf_(rbf)
    , n_poly_basis_(polynomial::basis_base::basis_size(poly_dimension, poly_degree)) {
    auto bbox = geometry::bbox3d::from_points(points);

    a_ = std::make_unique<fmm::fmm_operator<Order>>(rbf, fmm::fmm_tree_height(points.rows()), bbox);

    if (n_poly_basis_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
    }

    set_points(points);
  }

  rbf_operator(const rbf::rbf& rbf, int poly_dimension, int poly_degree,
               int tree_height, const geometry::bbox3d& bbox)
    : rbf_(rbf)
    , n_poly_basis_(polynomial::basis_base::basis_size(poly_dimension, poly_degree))
    , n_points_(0) {
    a_ = std::make_unique<fmm::fmm_operator<Order>>(rbf, tree_height, bbox);

    if (n_poly_basis_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(poly_dimension, poly_degree);
    }
  }

  common::valuesd operator()(const common::valuesd& weights) const override {
    assert(weights.rows() == size());

    common::valuesd y = common::valuesd::Zero(size());

    auto& rbf_kern = rbf_.get();
    auto diagonal = rbf_kern.evaluate(0.0) + rbf_kern.nugget();
    y.head(n_points_) = diagonal * weights.head(n_points_);

    a_->set_weights(weights.head(n_points_));
    y.head(n_points_) += a_->evaluate();

    if (n_poly_basis_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate(weights);
    }

    return y;
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = points.rows();

    a_->set_points(points);

    if (n_poly_basis_ > 0) {
      p_->set_points(points);
    }
  }

  size_t size() const override {
    return n_points_ + n_poly_basis_;
  }

private:
  const rbf::rbf rbf_;
  const size_t n_poly_basis_;

  size_t n_points_;
  std::unique_ptr<fmm::fmm_operator<Order>> a_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace interpolation
}  // namespace polatory
