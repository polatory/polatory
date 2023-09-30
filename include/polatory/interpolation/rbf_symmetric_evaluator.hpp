#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/precision.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <class Model>
class rbf_symmetric_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_symmetric_evaluator(const Model& model, const geometry::points3d& points,
                          const geometry::points3d& grad_points, precision prec)
      : dim_(model.poly_dimension()),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()) {
    auto bbox = geometry::bbox3d::from_points(points).convex_hull(
        geometry::bbox3d::from_points(grad_points));

    switch (dim_) {
      case 1:
        a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Model, 1>>(model, bbox, prec);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, 1>>(model, bbox, prec);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, 1>>(model, bbox, prec);
        h_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Model, 1>>(model, bbox, prec);
        break;
      case 2:
        a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Model, 2>>(model, bbox, prec);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, 2>>(model, bbox, prec);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, 2>>(model, bbox, prec);
        h_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Model, 2>>(model, bbox, prec);
        break;
      case 3:
        a_ = std::make_unique<fmm::fmm_symmetric_evaluator<Model, 3>>(model, bbox, prec);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, 3>>(model, bbox, prec);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, 3>>(model, bbox, prec);
        h_ = std::make_unique<fmm::fmm_hessian_symmetric_evaluator<Model, 3>>(model, bbox, prec);
        break;
    }

    a_->set_points(points);
    f_->set_source_points(grad_points);
    f_->set_field_points(points);
    ft_->set_source_points(points);
    ft_->set_field_points(grad_points);
    h_->set_points(grad_points);

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
      p_->set_field_points(points, grad_points);
    }
  }

  common::valuesd evaluate() const {
    common::valuesd y = common::valuesd::Zero(mu_ + dim_ * sigma_);

    y.head(mu_) += a_->evaluate();
    y.head(mu_) += f_->evaluate();
    y.tail(dim_ * sigma_) += ft_->evaluate();
    y.tail(dim_ * sigma_) += h_->evaluate();

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + dim_ * sigma_ + l_);

    a_->set_weights(weights.head(mu_));
    f_->set_weights(weights.segment(mu_, dim_ * sigma_));
    ft_->set_weights(weights.head(mu_));
    h_->set_weights(weights.segment(mu_, dim_ * sigma_));

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const int dim_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;

  std::unique_ptr<fmm::fmm_base_symmetric_evaluator> a_;
  std::unique_ptr<fmm::fmm_base_evaluator> f_;
  std::unique_ptr<fmm::fmm_base_evaluator> ft_;
  std::unique_ptr<fmm::fmm_base_symmetric_evaluator> h_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
