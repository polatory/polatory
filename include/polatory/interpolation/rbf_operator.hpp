#pragma once

#include <algorithm>
#include <limits>
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
#include <vector>

namespace polatory::interpolation {

template <int Dim>
class rbf_operator : public krylov::linear_operator {
  static constexpr int kDim = Dim;
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Bbox = geometry::bboxNd<kDim>;
  using FmmGenericEvaluatorPtr = fmm::FmmGenericEvaluatorPtr<kDim>;
  using FmmGenericSymmetricEvaluatorPtr = fmm::FmmGenericSymmetricEvaluatorPtr<kDim>;
  using Model = model<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  rbf_operator(const Model& model, const Points& points, const Points& grad_points,
               double accuracy = kInfinity, double grad_accuracy = kInfinity)
      : rbf_operator(model, Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points)),
                     accuracy, grad_accuracy) {
    set_points(points, grad_points);
  }

  rbf_operator(const Model& model, const Bbox& bbox, double accuracy = kInfinity,
               double grad_accuracy = kInfinity)
      : model_(model), l_(model.poly_basis_size()) {
    for (const auto& rbf : model.rbfs()) {
      a_.push_back(fmm::make_fmm_symmetric_evaluator(rbf, bbox, accuracy));
      f_.push_back(fmm::make_fmm_gradient_evaluator(rbf, bbox, accuracy));
      ft_.push_back(fmm::make_fmm_gradient_transpose_evaluator(rbf, bbox, grad_accuracy));
      h_.push_back(fmm::make_fmm_hessian_symmetric_evaluator(rbf, bbox, grad_accuracy));
    }

    if (l_ > 0) {
      poly_basis_ = std::make_unique<MonomialBasis>(model.poly_degree());
    }
  }

  vectord operator()(const vectord& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    vectord y = vectord::Zero(size());

    y.head(mu_) = weights.head(mu_) * model_.nugget();

    for (std::size_t i = 0; i < a_.size(); ++i) {
      a_.at(i)->set_weights(weights.head(mu_));
      f_.at(i)->set_weights(weights.segment(mu_, kDim * sigma_));
      ft_.at(i)->set_weights(weights.head(mu_));
      h_.at(i)->set_weights(weights.segment(mu_, kDim * sigma_));

      y.head(mu_) += a_.at(i)->evaluate();
      y.head(mu_) += f_.at(i)->evaluate();
      y.segment(mu_, kDim * sigma_) += ft_.at(i)->evaluate();
      y.segment(mu_, kDim * sigma_) += h_.at(i)->evaluate();
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y.head(mu_ + kDim * sigma_) += p_ * weights.tail(l_);
      y.tail(l_) += p_.transpose() * weights.head(mu_ + kDim * sigma_);
    }

    return y;
  }

  void set_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    for (std::size_t i = 0; i < a_.size(); ++i) {
      a_.at(i)->set_points(points);
      f_.at(i)->set_source_points(grad_points);
      f_.at(i)->set_target_points(points);
      ft_.at(i)->set_source_points(points);
      ft_.at(i)->set_target_points(grad_points);
      h_.at(i)->set_points(grad_points);
    }

    if (l_ > 0) {
      p_ = poly_basis_->evaluate(points, grad_points);
    }
  }

  index_t size() const override { return mu_ + kDim * sigma_ + l_; }

 private:
  const Model& model_;
  const index_t l_;
  index_t mu_{};
  index_t sigma_{};

  std::vector<FmmGenericSymmetricEvaluatorPtr> a_;
  std::vector<FmmGenericEvaluatorPtr> f_;
  std::vector<FmmGenericEvaluatorPtr> ft_;
  std::vector<FmmGenericSymmetricEvaluatorPtr> h_;
  std::unique_ptr<MonomialBasis> poly_basis_;
  matrixd p_;
};

}  // namespace polatory::interpolation
