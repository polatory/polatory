#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <class Model, int Order = 10>
struct rbf_operator : krylov::linear_operator {
  rbf_operator(const Model& model, const geometry::points3d& points,
               const geometry::points3d& grad_points)
      : rbf_operator(model, geometry::bbox3d::from_points(points).convex_hull(
                                geometry::bbox3d::from_points(grad_points))) {
    set_points(points, grad_points);
  }

  rbf_operator(const Model& model, const geometry::bbox3d& bbox)
      : model_(model), dim_(model.poly_dimension()), l_(model.poly_basis_size()) {
    switch (dim_) {
      case 1:
        a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Model, Order, 1>>(model, bbox);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, Order, 1>>(model, bbox);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, Order, 1>>(model, bbox);
        h_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Model, Order, 1>>(model, bbox);
        break;
      case 2:
        a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Model, Order, 2>>(model, bbox);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, Order, 2>>(model, bbox);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, Order, 2>>(model, bbox);
        h_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Model, Order, 2>>(model, bbox);
        break;
      case 3:
        a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Model, Order, 3>>(model, bbox);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, Order, 3>>(model, bbox);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, Order, 3>>(model, bbox);
        h_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Model, Order, 3>>(model, bbox);
        break;
    }

    if (l_ > 0) {
      poly_basis_ =
          std::make_unique<polynomial::monomial_basis>(model.poly_dimension(), model.poly_degree());
    }
  }

  common::valuesd operator()(const common::valuesd& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    common::valuesd y = common::valuesd::Zero(size());

    a_->set_weights(weights.head(mu_));
    f_->set_weights(weights.segment(mu_, dim_ * sigma_));
    ft_->set_weights(weights.head(mu_));
    h_->set_weights(weights.segment(mu_, dim_ * sigma_));

    y.head(mu_) += a_->evaluate();
    y.head(mu_) += f_->evaluate();
    y.segment(mu_, dim_ * sigma_) += ft_->evaluate();
    y.segment(mu_, dim_ * sigma_) += h_->evaluate();

    if (l_ > 0) {
      // Add polynomial terms.
      y.head(mu_ + dim_ * sigma_) += pt_.transpose() * weights.tail(l_);
      y.tail(l_) += pt_ * weights.head(mu_ + dim_ * sigma_);
    }

    y.head(mu_) += weights.head(mu_) * model_.nugget();

    return y;
  }

  void set_points(const geometry::points3d& points, const geometry::points3d& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    a_->set_points(points);
    f_->set_source_points(grad_points);
    f_->set_field_points(points);
    ft_->set_source_points(points);
    ft_->set_field_points(grad_points);
    h_->set_points(grad_points);

    if (l_ > 0) {
      pt_ = poly_basis_->evaluate(points, grad_points);
    }
  }

  index_t size() const override { return mu_ + dim_ * sigma_ + l_; }

 private:
  const Model& model_;
  const int dim_;
  const index_t l_;
  index_t mu_{};
  index_t sigma_{};

  std::unique_ptr<fmm::fmm_base_symmetric_evaluator> a_;
  std::unique_ptr<fmm::fmm_base_evaluator> f_;
  std::unique_ptr<fmm::fmm_base_evaluator> ft_;
  std::unique_ptr<fmm::fmm_base_symmetric_evaluator> h_;
  std::unique_ptr<polynomial::monomial_basis> poly_basis_;
  Eigen::MatrixXd pt_;
};

}  // namespace polatory::interpolation
