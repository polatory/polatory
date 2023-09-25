#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_gradient_evaluator.hpp>
#include <polatory/fmm/fmm_gradient_transpose_evaluator.hpp>
#include <polatory/fmm/fmm_hessian_symmetric_evaluator.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Order = 10>
class rbf_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_symmetric_evaluator(const model& model, const geometry::points3d& points)
      : rbf_symmetric_evaluator(model, points, geometry::points3d(0, 3)) {}

  rbf_symmetric_evaluator(const model& model, const geometry::points3d& points,
                          const geometry::points3d& grad_points)
      : dim_(model.poly_dimension()),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()) {
    auto bbox = geometry::bbox3d::from_points(points).convex_hull(
        geometry::bbox3d::from_points(grad_points));

    if (mu_ > 0) {
      a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Order>>(model, fmm::fmm_tree_height(mu_),
                                                                 bbox);
      a_->set_points(points);
    }

    if (mu_ > 0 && sigma_ > 0) {
      switch (dim_) {
        case 1:
          f1_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 1>>(
              model, fmm::fmm_tree_height(std::max(mu_, sigma_)), bbox);
          f1_->set_source_points(grad_points);
          f1_->set_field_points(points);
          ft1_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 1>>(
              model, fmm::fmm_tree_height(std::max(mu_, sigma_)), bbox);
          ft1_->set_source_points(points);
          ft1_->set_field_points(grad_points);
          break;
        case 2:
          f2_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 2>>(
              model, fmm::fmm_tree_height(std::max(mu_, sigma_)), bbox);
          f2_->set_source_points(grad_points);
          f2_->set_field_points(points);
          ft2_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 2>>(
              model, fmm::fmm_tree_height(std::max(mu_, sigma_)), bbox);
          ft2_->set_source_points(points);
          ft2_->set_field_points(grad_points);
          break;
        case 3:
          f3_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 3>>(
              model, fmm::fmm_tree_height(std::max(mu_, sigma_)), bbox);
          f3_->set_source_points(grad_points);
          f3_->set_field_points(points);
          ft3_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 3>>(
              model, fmm::fmm_tree_height(std::max(mu_, sigma_)), bbox);
          ft3_->set_source_points(points);
          ft3_->set_field_points(grad_points);
          break;
      }
    }

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          h1_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Order, 1>>(
              model, fmm::fmm_tree_height(sigma_), bbox);
          h1_->set_points(grad_points);
          break;
        case 2:
          h2_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Order, 2>>(
              model, fmm::fmm_tree_height(sigma_), bbox);
          h2_->set_points(grad_points);
          break;
        case 3:
          h3_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Order, 3>>(
              model, fmm::fmm_tree_height(sigma_), bbox);
          h3_->set_points(grad_points);
          break;
      }
    }

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
      p_->set_field_points(points, grad_points);
    }
  }

  common::valuesd evaluate() const {
    common::valuesd y = common::valuesd::Zero(mu_ + dim_ * sigma_);

    if (mu_ > 0) {
      y.head(mu_) += a_->evaluate();
    }

    if (mu_ > 0 && sigma_ > 0) {
      switch (dim_) {
        case 1:
          y.head(mu_) += f1_->evaluate();
          y.tail(dim_ * sigma_) += ft1_->evaluate();
          break;
        case 2:
          y.head(mu_) += f2_->evaluate();
          y.tail(dim_ * sigma_) += ft2_->evaluate();
          break;
        case 3:
          y.head(mu_) += f3_->evaluate();
          y.tail(dim_ * sigma_) += ft3_->evaluate();
          break;
      }
    }

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          y.tail(dim_ * sigma_) += h1_->evaluate();
          break;
        case 2:
          y.tail(dim_ * sigma_) += h2_->evaluate();
          break;
        case 3:
          y.tail(dim_ * sigma_) += h3_->evaluate();
          break;
      }
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + dim_ * sigma_ + l_);

    if (mu_ > 0) {
      a_->set_weights(weights.head(mu_));
    }

    if (mu_ > 0 && sigma_ > 0) {
      switch (dim_) {
        case 1:
          f1_->set_weights(weights.segment(mu_, dim_ * sigma_));
          ft1_->set_weights(weights.head(mu_));
          break;
        case 2:
          f2_->set_weights(weights.segment(mu_, dim_ * sigma_));
          ft2_->set_weights(weights.head(mu_));
          break;
        case 3:
          f3_->set_weights(weights.segment(mu_, dim_ * sigma_));
          ft3_->set_weights(weights.head(mu_));
          break;
      }
    }

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          h1_->set_weights(weights.segment(mu_, dim_ * sigma_));
          break;
        case 2:
          h2_->set_weights(weights.segment(mu_, dim_ * sigma_));
          break;
        case 3:
          h3_->set_weights(weights.segment(mu_, dim_ * sigma_));
          break;
      }
    }

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const int dim_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;

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
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
