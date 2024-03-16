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
#include <vector>

namespace polatory::interpolation {

template <int Dim>
class rbf_operator : public krylov::linear_operator {
  static constexpr int kDim = Dim;
  using Bbox = geometry::bboxNd<kDim>;
  using FmmGenericEvaluatorPtr = fmm::FmmGenericEvaluatorPtr<kDim>;
  using FmmGenericSymmetricEvaluatorPtr = fmm::FmmGenericSymmetricEvaluatorPtr<kDim>;
  using Model = model<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  rbf_operator(const Model& model, const Points& points, const Points& grad_points, int order)
      : rbf_operator(model, Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points)),
                     order) {
    set_points(points, grad_points);
  }

  rbf_operator(const Model& model, const Bbox& bbox, int order)
      : model_(model), l_(model.poly_basis_size()) {
    for (const auto& rbf : model.rbfs()) {
      a_.push_back(fmm::make_fmm_symmetric_evaluator(rbf, bbox, order));
      f_.push_back(fmm::make_fmm_gradient_evaluator(rbf, bbox, order));
      ft_.push_back(fmm::make_fmm_gradient_transpose_evaluator(rbf, bbox, order));
      h_.push_back(fmm::make_fmm_hessian_symmetric_evaluator(rbf, bbox, order));
    }

    if (l_ > 0) {
      poly_basis_ = std::make_unique<MonomialBasis>(model.poly_degree());
    }
  }

  common::valuesd operator()(const common::valuesd& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    common::valuesd y = common::valuesd::Zero(size());

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
      y.head(mu_ + kDim * sigma_) += pt_.transpose() * weights.tail(l_);
      y.tail(l_) += pt_ * weights.head(mu_ + kDim * sigma_);
    }

    y.head(mu_) += weights.head(mu_) * model_.nugget();

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
      pt_ = poly_basis_->evaluate(points, grad_points);
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
  Eigen::MatrixXd pt_;
};

}  // namespace polatory::interpolation
