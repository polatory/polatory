#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/fmm/fmm_gradient_evaluator.hpp>
#include <polatory/fmm/fmm_gradient_transpose_evaluator.hpp>
#include <polatory/fmm/fmm_hessian_evaluator.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Order = 10>
class rbf_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_evaluator(const model& model, const geometry::points3d& source_points)
      : rbf_evaluator(model, source_points, geometry::points3d(0, 3)) {}

  rbf_evaluator(const model& model, const geometry::points3d& source_points,
                const geometry::points3d& source_grad_points)
      : rbf_evaluator(model, source_points, source_grad_points,
                      geometry::bbox3d::from_points(source_points)
                          .convex_hull(geometry::bbox3d::from_points(source_grad_points))) {}

  rbf_evaluator(const model& model, const geometry::points3d& source_points,
                const geometry::bbox3d& bbox)
      : rbf_evaluator(model, source_points, geometry::points3d(0, 3), bbox) {}

  rbf_evaluator(const model& model, const geometry::points3d& source_points,
                const geometry::points3d& source_grad_points, const geometry::bbox3d& bbox)
      : l_(model.poly_basis_size()), dim_(model.poly_dimension()) {
    auto mu = source_points.rows();
    a_ = std::make_unique<fmm::fmm_evaluator<Order>>(model, fmm::fmm_tree_height(mu), bbox);

    auto sigma = source_grad_points.rows();
    if (sigma > 0) {
      switch (dim_) {
        case 1:
          f1_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 1>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          ft1_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 1>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          h1_ = std::make_unique<fmm::fmm_hessian_evaluator<Order, 1>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          break;
        case 2:
          f2_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 2>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          ft2_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 2>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          h2_ = std::make_unique<fmm::fmm_hessian_evaluator<Order, 2>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          break;
        case 3:
          f3_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 3>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          ft3_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 3>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          h3_ = std::make_unique<fmm::fmm_hessian_evaluator<Order, 3>>(
              model, fmm::fmm_tree_height(sigma), bbox);
          break;
      }
    }

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
    }

    set_source_points(source_points, source_grad_points);
  }

  rbf_evaluator(const model& model, int tree_height, const geometry::bbox3d& bbox)
      : l_(model.poly_basis_size()), dim_(model.poly_dimension()) {
    a_ = std::make_unique<fmm::fmm_evaluator<Order>>(model, tree_height, bbox);

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          f1_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 1>>(model, tree_height, bbox);
          ft1_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 1>>(
              model, tree_height, bbox);
          h1_ = std::make_unique<fmm::fmm_hessian_evaluator<Order, 1>>(model, tree_height, bbox);
          break;
        case 2:
          f2_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 2>>(model, tree_height, bbox);
          ft2_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 2>>(
              model, tree_height, bbox);
          h2_ = std::make_unique<fmm::fmm_hessian_evaluator<Order, 2>>(model, tree_height, bbox);
          break;
        case 3:
          f3_ = std::make_unique<fmm::fmm_gradient_evaluator<Order, 3>>(model, tree_height, bbox);
          ft3_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Order, 3>>(
              model, tree_height, bbox);
          h3_ = std::make_unique<fmm::fmm_hessian_evaluator<Order, 3>>(model, tree_height, bbox);
          break;
      }
    }

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
    }
  }

  common::valuesd evaluate() const {
    common::valuesd y = common::valuesd::Zero(fld_mu_ + dim_ * fld_sigma_);

    y.head(fld_mu_) += a_->evaluate();

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          y.head(fld_mu_) += f1_->evaluate();
          y.tail(dim_ * fld_sigma_) += ft1_->evaluate();
          y.tail(dim_ * fld_sigma_) += h1_->evaluate();
          break;
        case 2:
          y.head(fld_mu_) += f2_->evaluate();
          y.tail(dim_ * fld_sigma_) += ft2_->evaluate();
          y.tail(dim_ * fld_sigma_) += h2_->evaluate();
          break;
        case 3:
          y.head(fld_mu_) += f3_->evaluate();
          y.tail(dim_ * fld_sigma_) += ft3_->evaluate();
          y.tail(dim_ * fld_sigma_) += h3_->evaluate();
          break;
      }
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  common::valuesd evaluate(const geometry::points3d& field_points) {
    return evaluate(field_points, geometry::points3d(0, 3));
  }

  common::valuesd evaluate(const geometry::points3d& field_points,
                           const geometry::points3d& field_grad_points) {
    set_field_points(field_points, field_grad_points);

    return evaluate();
  }

  void set_field_points(const geometry::points3d& points) {
    set_field_points(points, geometry::points3d(0, 3));
  }

  void set_field_points(const geometry::points3d& points, const geometry::points3d& grad_points) {
    fld_mu_ = points.rows();
    fld_sigma_ = grad_points.rows();

    a_->set_field_points(points);

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          f1_->set_field_points(points);
          ft1_->set_field_points(grad_points);
          h1_->set_field_points(grad_points);
          break;
        case 2:
          f2_->set_field_points(points);
          ft2_->set_field_points(grad_points);
          h2_->set_field_points(grad_points);
          break;
        case 3:
          f3_->set_field_points(points);
          ft3_->set_field_points(grad_points);
          h3_->set_field_points(grad_points);
          break;
      }
    }

    if (l_ > 0) {
      p_->set_field_points(points, grad_points);
    }
  }

  void set_source_points(const geometry::points3d& points) {
    set_source_points(points, geometry::points3d(0, 3));
  }

  void set_source_points(const geometry::points3d& points, const geometry::points3d& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    a_->set_source_points(points);

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          f1_->set_source_points(grad_points);
          ft1_->set_source_points(points);
          h1_->set_source_points(grad_points);
          break;
        case 2:
          f2_->set_source_points(grad_points);
          ft2_->set_source_points(points);
          h2_->set_source_points(grad_points);
          break;
        case 3:
          f3_->set_source_points(grad_points);
          ft3_->set_source_points(points);
          h3_->set_source_points(grad_points);
          break;
      }
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + dim_ * sigma_ + l_);

    a_->set_weights(weights.head(mu_));

    if (sigma_ > 0) {
      switch (dim_) {
        case 1:
          f1_->set_weights(weights.segment(mu_, dim_ * sigma_));
          ft1_->set_weights(weights.head(mu_));
          h1_->set_weights(weights.segment(mu_, dim_ * sigma_));
          break;
        case 2:
          f2_->set_weights(weights.segment(mu_, dim_ * sigma_));
          ft2_->set_weights(weights.head(mu_));
          h2_->set_weights(weights.segment(mu_, dim_ * sigma_));
          break;
        case 3:
          f3_->set_weights(weights.segment(mu_, dim_ * sigma_));
          ft3_->set_weights(weights.head(mu_));
          h3_->set_weights(weights.segment(mu_, dim_ * sigma_));
          break;
      }
    }

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const index_t l_;
  const index_t dim_;
  index_t mu_{};
  index_t sigma_{};
  index_t fld_mu_{};
  index_t fld_sigma_{};
  std::unique_ptr<fmm::fmm_evaluator<Order>> a_;
  std::unique_ptr<fmm::fmm_gradient_evaluator<Order, 1>> f1_;
  std::unique_ptr<fmm::fmm_gradient_evaluator<Order, 2>> f2_;
  std::unique_ptr<fmm::fmm_gradient_evaluator<Order, 3>> f3_;
  std::unique_ptr<fmm::fmm_gradient_transpose_evaluator<Order, 1>> ft1_;
  std::unique_ptr<fmm::fmm_gradient_transpose_evaluator<Order, 2>> ft2_;
  std::unique_ptr<fmm::fmm_gradient_transpose_evaluator<Order, 3>> ft3_;
  std::unique_ptr<fmm::fmm_hessian_evaluator<Order, 1>> h1_;
  std::unique_ptr<fmm::fmm_hessian_evaluator<Order, 2>> h2_;
  std::unique_ptr<fmm::fmm_hessian_evaluator<Order, 3>> h3_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
