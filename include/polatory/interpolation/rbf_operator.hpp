#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_gradient_evaluator.hpp>
#include <polatory/fmm/fmm_gradient_transpose_evaluator.hpp>
#include <polatory/fmm/fmm_hessian_symmetric_evaluator.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Order = 10>
struct rbf_operator : krylov::linear_operator {
  rbf_operator(const model& model, const geometry::points3d& points)
      : rbf_operator(model, points, geometry::points3d(0, 3)) {}

  rbf_operator(const model& model, const geometry::points3d& points,
               const geometry::points3d& grad_points)
      : model_(model), dim_(model.poly_dimension()), l_(model.poly_basis_size()) {
    auto mu = points.rows();
    auto sigma = grad_points.rows();
    auto bbox = geometry::bbox3d::from_points(points).convex_hull(
        geometry::bbox3d::from_points(grad_points));

    a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Order>>(model, fmm::fmm_tree_height(mu),
                                                               bbox);

    switch (dim_) {
      case 1:
        f1_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 1>>(
            model, fmm::fmm_tree_height(std::max(mu, sigma)), bbox);
        ft1_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 1>>(
            model, fmm::fmm_tree_height(std::max(mu, sigma)), bbox);
        h1_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Order, 1>>(
            model, fmm::fmm_tree_height(sigma), bbox);
        break;
      case 2:
        f2_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 2>>(
            model, fmm::fmm_tree_height(std::max(mu, sigma)), bbox);
        ft2_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 2>>(
            model, fmm::fmm_tree_height(std::max(mu, sigma)), bbox);
        h2_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Order, 2>>(
            model, fmm::fmm_tree_height(sigma), bbox);
        break;
      case 3:
        f3_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 3>>(
            model, fmm::fmm_tree_height(std::max(mu, sigma)), bbox);
        ft3_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 3>>(
            model, fmm::fmm_tree_height(std::max(mu, sigma)), bbox);
        h3_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Order, 3>>(
            model, fmm::fmm_tree_height(sigma), bbox);
        break;
    }

    if (l_ > 0) {
      poly_basis_ =
          std::make_unique<polynomial::monomial_basis>(model.poly_dimension(), model.poly_degree());
    }

    set_points(points, grad_points);
  }

  rbf_operator(const model& model, int tree_height, const geometry::bbox3d& bbox)
      : model_(model),
        dim_(model.poly_dimension()),
        l_(model.poly_basis_size()),
        a_(std::make_unique<fmm::fmm_symmetric_evaluator<Order>>(model, tree_height, bbox)) {
    if (l_ > 0) {
      poly_basis_ =
          std::make_unique<polynomial::monomial_basis>(model.poly_dimension(), model.poly_degree());
    }
  }

  common::valuesd operator()(const common::valuesd& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    common::valuesd y = common::valuesd::Zero(size());

    a_->set_weights(weights.head(mu_));
    y.head(mu_) = a_->evaluate();

    switch (dim_) {
      case 1:
        f1_->set_weights(weights.segment(mu_, dim_ * sigma_));
        y.head(mu_) += f1_->evaluate();
        ft1_->set_weights(weights.head(mu_));
        y.segment(mu_, dim_ * sigma_) += ft1_->evaluate();
        h1_->set_weights(weights.segment(mu_, dim_ * sigma_));
        y.segment(mu_, dim_ * sigma_) += h1_->evaluate();
        break;
      case 2:
        f2_->set_weights(weights.segment(mu_, dim_ * sigma_));
        y.head(mu_) += f2_->evaluate();
        ft2_->set_weights(weights.head(mu_));
        y.segment(mu_, dim_ * sigma_) += ft2_->evaluate();
        h2_->set_weights(weights.segment(mu_, dim_ * sigma_));
        y.segment(mu_, dim_ * sigma_) += h2_->evaluate();
        break;
      case 3:
        f3_->set_weights(weights.segment(mu_, dim_ * sigma_));
        y.head(mu_) += f3_->evaluate();
        ft3_->set_weights(weights.head(mu_));
        y.segment(mu_, dim_ * sigma_) += ft3_->evaluate();
        h3_->set_weights(weights.segment(mu_, dim_ * sigma_));
        y.segment(mu_, dim_ * sigma_) += h3_->evaluate();
        break;
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y.head(mu_ + dim_ * sigma_) += pt_.transpose() * weights.tail(l_);
      y.tail(l_) += pt_ * weights.head(mu_ + dim_ * sigma_);
    }

    y.head(mu_) += weights.head(mu_) * model_.nugget();

    return y;
  }

  void set_points(const geometry::points3d& points) {
    set_points(points, geometry::points3d(0, 3));
  }

  void set_points(const geometry::points3d& points, const geometry::points3d& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    a_->set_points(points);

    switch (dim_) {
      case 1:
        f1_->set_source_points(grad_points);
        f1_->set_field_points(points);
        ft1_->set_source_points(points);
        ft1_->set_field_points(grad_points);
        h1_->set_points(grad_points);
        break;
      case 2:
        f2_->set_source_points(grad_points);
        f2_->set_field_points(points);
        ft2_->set_source_points(points);
        ft2_->set_field_points(grad_points);
        h2_->set_points(grad_points);
        break;
      case 3:
        f3_->set_source_points(grad_points);
        f3_->set_field_points(points);
        ft3_->set_source_points(points);
        ft3_->set_field_points(grad_points);
        h3_->set_points(grad_points);
        break;
    }

    if (l_ > 0) {
      pt_ = poly_basis_->evaluate(points, grad_points);
    }
  }

  index_t size() const override { return mu_ + dim_ * sigma_ + l_; }

 private:
  const model& model_;
  const index_t dim_;
  const index_t l_;
  index_t mu_{};
  index_t sigma_{};

  std::unique_ptr<fmm::fmm_symmetric_evaluator<Order>> a_;
  std::unique_ptr<fmm::fmm_gradient_evaluator<Order, 1>> f1_;
  std::unique_ptr<fmm::fmm_gradient_evaluator<Order, 2>> f2_;
  std::unique_ptr<fmm::fmm_gradient_evaluator<Order, 3>> f3_;
  std::unique_ptr<fmm::fmm_gradient_transpose_evaluator<Order, 1>> ft1_;
  std::unique_ptr<fmm::fmm_gradient_transpose_evaluator<Order, 2>> ft2_;
  std::unique_ptr<fmm::fmm_gradient_transpose_evaluator<Order, 3>> ft3_;
  std::unique_ptr<fmm::fmm_hessian_symmetric_evaluator<Order, 1>> h1_;
  std::unique_ptr<fmm::fmm_hessian_symmetric_evaluator<Order, 2>> h2_;
  std::unique_ptr<fmm::fmm_hessian_symmetric_evaluator<Order, 3>> h3_;
  std::unique_ptr<polynomial::monomial_basis> poly_basis_;
  Eigen::MatrixXd pt_;
};

}  // namespace polatory::interpolation
