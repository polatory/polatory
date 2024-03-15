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
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Dim>
class rbf_symmetric_evaluator {
  static constexpr int kDim = Dim;
  using Bbox = geometry::bboxNd<kDim>;
  using FmmGenericEvaluatorPtr = fmm::FmmGenericEvaluatorPtr<kDim>;
  using FmmGenericSymmetricEvaluatorPtr = fmm::FmmGenericSymmetricEvaluatorPtr<kDim>;
  using Model = model<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using PolynomialEvaluator = polynomial::polynomial_evaluator<MonomialBasis>;

 public:
  rbf_symmetric_evaluator(const Model& model, const Points& points, const Points& grad_points,
                          int order)
      : rbf_symmetric_evaluator(
            model, Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points)), order) {
    set_points(points, grad_points);
  }

  rbf_symmetric_evaluator(const Model& model, const Bbox& bbox, int order)
      : l_(model.poly_basis_size()),
        a_(fmm::make_fmm_symmetric_evaluator(model.rbf(), bbox, order)),
        f_(fmm::make_fmm_gradient_evaluator(model.rbf(), bbox, order)),
        ft_(fmm::make_fmm_gradient_transpose_evaluator(model.rbf(), bbox, order)),
        h_(fmm::make_fmm_hessian_symmetric_evaluator(model.rbf(), bbox, order)) {
    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  common::valuesd evaluate() const {
    common::valuesd y = common::valuesd::Zero(mu_ + kDim * sigma_);

    y.head(mu_) += a_->evaluate();
    y.head(mu_) += f_->evaluate();
    y.tail(kDim * sigma_) += ft_->evaluate();
    y.tail(kDim * sigma_) += h_->evaluate();

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    a_->set_points(points);
    f_->set_source_points(grad_points);
    f_->set_target_points(points);
    ft_->set_source_points(points);
    ft_->set_target_points(grad_points);
    h_->set_points(grad_points);

    if (l_ > 0) {
      p_->set_target_points(points, grad_points);
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    a_->set_weights(weights.head(mu_));
    f_->set_weights(weights.segment(mu_, kDim * sigma_));
    ft_->set_weights(weights.head(mu_));
    h_->set_weights(weights.segment(mu_, kDim * sigma_));

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const index_t l_;
  index_t mu_{};
  index_t sigma_{};

  FmmGenericSymmetricEvaluatorPtr a_;
  FmmGenericEvaluatorPtr f_;
  FmmGenericEvaluatorPtr ft_;
  FmmGenericSymmetricEvaluatorPtr h_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
