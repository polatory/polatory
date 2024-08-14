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
class Operator : public krylov::LinearOperator {
  static constexpr int kDim = Dim;
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Bbox = geometry::Bbox<kDim>;
  using FmmGenericEvaluatorPtr = fmm::FmmGenericEvaluatorPtr<kDim>;
  using FmmGenericSymmetricEvaluatorPtr = fmm::FmmGenericSymmetricEvaluatorPtr<kDim>;
  using Model = Model<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  Operator(const Model& model, const Points& points, const Points& grad_points,
           double accuracy = kInfinity, double grad_accuracy = kInfinity)
      : Operator(model, Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points)),
                 accuracy, grad_accuracy) {
    set_points(points, grad_points);
  }

  Operator(const Model& model, const Bbox& bbox, double accuracy = kInfinity,
           double grad_accuracy = kInfinity)
      : model_(model),
        l_(model.poly_basis_size()),
        accuracy_(accuracy),
        grad_accuracy_(grad_accuracy) {
    for (const auto& rbf : model.rbfs()) {
      a_.push_back(fmm::make_fmm_symmetric_evaluator(rbf, bbox));
      f_.push_back(fmm::make_fmm_gradient_evaluator(rbf, bbox));
      ft_.push_back(fmm::make_fmm_gradient_transpose_evaluator(rbf, bbox));
      h_.push_back(fmm::make_fmm_hessian_symmetric_evaluator(rbf, bbox));
    }

    if (l_ > 0) {
      poly_basis_ = std::make_unique<MonomialBasis>(model.poly_degree());
    }
  }

  VecX operator()(const VecX& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    VecX y = VecX::Zero(size());

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

    auto accuracy = (sigma_ > 0 ? accuracy_ / 2.0 : accuracy_) / static_cast<double>(a_.size());
    auto grad_accuracy =
        (sigma_ > 0 ? grad_accuracy_ / 2.0 : grad_accuracy_) / static_cast<double>(a_.size());

    for (std::size_t i = 0; i < a_.size(); ++i) {
      a_.at(i)->set_points(points);
      f_.at(i)->set_source_points(grad_points);
      f_.at(i)->set_target_points(points);
      ft_.at(i)->set_source_points(points);
      ft_.at(i)->set_target_points(grad_points);
      h_.at(i)->set_points(grad_points);

      a_.at(i)->set_accuracy(accuracy);
      f_.at(i)->set_accuracy(accuracy);
      ft_.at(i)->set_accuracy(grad_accuracy);
      h_.at(i)->set_accuracy(grad_accuracy);
    }

    if (l_ > 0) {
      p_ = poly_basis_->evaluate(points, grad_points);
    }
  }

  Index size() const override { return mu_ + kDim * sigma_ + l_; }

 private:
  const Model& model_;
  const Index l_;
  const double accuracy_;
  const double grad_accuracy_;
  Index mu_{};
  Index sigma_{};

  std::vector<FmmGenericSymmetricEvaluatorPtr> a_;
  std::vector<FmmGenericEvaluatorPtr> f_;
  std::vector<FmmGenericEvaluatorPtr> ft_;
  std::vector<FmmGenericSymmetricEvaluatorPtr> h_;
  std::unique_ptr<MonomialBasis> poly_basis_;
  MatX p_;
};

}  // namespace polatory::interpolation
