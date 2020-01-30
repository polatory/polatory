// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace interpolation {

template <int Order = 10>
struct rbf_operator : krylov::linear_operator {
  rbf_operator(const model& model, const geometry::points3d& points)
    : model_(model)
    , n_poly_basis_(model.poly_basis_size())
    , n_points_(0) {
    auto n_points = static_cast<index_t>(points.rows());
    auto bbox = geometry::bbox3d::from_points(points);
    a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Order>>(model, fmm::fmm_tree_height(n_points), bbox);

    if (n_poly_basis_ > 0) {
      poly_basis_ = std::make_unique<polynomial::monomial_basis>(model.poly_dimension(), model.poly_degree());
    }

    set_points(points);
  }

  rbf_operator(const model& model, int tree_height, const geometry::bbox3d& bbox)
    : model_(model)
    , n_poly_basis_(model.poly_basis_size())
    , n_points_(0) {
    a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Order>>(model, tree_height, bbox);

    if (n_poly_basis_ > 0) {
      poly_basis_ = std::make_unique<polynomial::monomial_basis>(model.poly_dimension(), model.poly_degree());
    }
  }

  common::valuesd operator()(const common::valuesd& weights) const override {
    POLATORY_ASSERT(static_cast<index_t>(weights.rows()) == size());

    common::valuesd y = common::valuesd::Zero(size());

    auto& rbf = model_.rbf();
    auto diagonal = rbf.evaluate_untransformed(0.0) + model_.nugget();
    y.head(n_points_) = diagonal * weights.head(n_points_);

    a_->set_weights(weights.head(n_points_));
    y.head(n_points_) += a_->evaluate();

    if (n_poly_basis_ > 0) {
      // Add polynomial terms.
      y.head(n_points_) += pt_.transpose() * weights.tail(n_poly_basis_);
      y.tail(n_poly_basis_) += pt_ * weights.head(n_points_);
    }

    return y;
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = static_cast<index_t>(points.rows());

    a_->set_points(points);

    if (n_poly_basis_ > 0) {
      pt_ = poly_basis_->evaluate(points);
    }
  }

  index_t size() const override {
    return n_points_ + n_poly_basis_;
  }

private:
  const model& model_;
  const index_t n_poly_basis_;

  index_t n_points_;
  std::unique_ptr<fmm::fmm_symmetric_evaluator<Order>> a_;
  std::unique_ptr<polynomial::monomial_basis> poly_basis_;
  Eigen::MatrixXd pt_;
};

}  // namespace interpolation
}  // namespace polatory
