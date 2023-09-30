#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <class Model, int Order = 10>
class rbf_evaluator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_evaluator(const Model& model, const geometry::points3d& source_points)
      : rbf_evaluator(model, source_points, geometry::points3d(0, 3)) {}

  rbf_evaluator(const Model& model, const geometry::points3d& source_points,
                const geometry::points3d& source_grad_points)
      : rbf_evaluator(model, source_points, source_grad_points,
                      geometry::bbox3d::from_points(source_points)
                          .convex_hull(geometry::bbox3d::from_points(source_grad_points))) {}

  rbf_evaluator(const Model& model, const geometry::points3d& source_points,
                const geometry::bbox3d& bbox)
      : rbf_evaluator(model, source_points, geometry::points3d(0, 3), bbox) {}

  rbf_evaluator(const Model& model, const geometry::points3d& source_points,
                const geometry::points3d& source_grad_points, const geometry::bbox3d& bbox)
      : rbf_evaluator(model, bbox) {
    set_source_points(source_points, source_grad_points);
  }

  rbf_evaluator(const Model& model, const geometry::bbox3d& bbox)
      : dim_(model.poly_dimension()), l_(model.poly_basis_size()) {
    switch (dim_) {
      case 1:
        a_ = std::make_unique<fmm::fmm_evaluator<Model, Order, 1>>(model, bbox);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, Order, 1>>(model, bbox);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, Order, 1>>(model, bbox);
        h_ = std::make_unique<fmm::fmm_hessian_evaluator<Model, Order, 1>>(model, bbox);
        break;
      case 2:
        a_ = std::make_unique<fmm::fmm_evaluator<Model, Order, 2>>(model, bbox);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, Order, 2>>(model, bbox);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, Order, 2>>(model, bbox);
        h_ = std::make_unique<fmm::fmm_hessian_evaluator<Model, Order, 2>>(model, bbox);
        break;
      case 3:
        a_ = std::make_unique<fmm::fmm_evaluator<Model, Order, 3>>(model, bbox);
        f_ = std::make_unique<fmm::fmm_gradient_evaluator<Model, Order, 3>>(model, bbox);
        ft_ = std::make_unique<fmm::fmm_gradient_transpose_evaluator<Model, Order, 3>>(model, bbox);
        h_ = std::make_unique<fmm::fmm_hessian_evaluator<Model, Order, 3>>(model, bbox);
        break;
    }

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
    }
  }

  common::valuesd evaluate() const {
    common::valuesd y = common::valuesd::Zero(fld_mu_ + dim_ * fld_sigma_);

    y.head(fld_mu_) += a_->evaluate();
    y.head(fld_mu_) += f_->evaluate();
    y.tail(dim_ * fld_sigma_) += ft_->evaluate();
    y.tail(dim_ * fld_sigma_) += h_->evaluate();

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
    f_->set_field_points(points);
    ft_->set_field_points(grad_points);
    h_->set_field_points(grad_points);

    if (l_ > 0) {
      p_->set_field_points(points, grad_points);
    }
  }

  void set_source_points(const geometry::points3d& points, const geometry::points3d& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    a_->set_source_points(points);
    f_->set_source_points(grad_points);
    ft_->set_source_points(points);
    h_->set_source_points(grad_points);
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
  index_t mu_{};
  index_t sigma_{};
  index_t fld_mu_{};
  index_t fld_sigma_{};
  std::unique_ptr<fmm::fmm_base_evaluator> a_;
  std::unique_ptr<fmm::fmm_base_evaluator> f_;
  std::unique_ptr<fmm::fmm_base_evaluator> ft_;
  std::unique_ptr<fmm::fmm_base_evaluator> h_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
