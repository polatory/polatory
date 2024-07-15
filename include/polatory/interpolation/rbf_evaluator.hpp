#pragma once

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::interpolation {

template <int Dim>
class rbf_evaluator {
  static constexpr int kDim = Dim;
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Bbox = geometry::bboxNd<kDim>;
  using FmmGenericEvaluatorPtr = fmm::FmmGenericEvaluatorPtr<kDim>;
  using Model = model<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using PolynomialEvaluator = polynomial::polynomial_evaluator<MonomialBasis>;

 public:
  rbf_evaluator(const Model& model, const Points& source_points, double accuracy = kInfinity)
      : rbf_evaluator(model, source_points, Points(0, kDim), accuracy, kInfinity) {}

  rbf_evaluator(const Model& model, const Points& source_points, const Points& source_grad_points,
                double accuracy = kInfinity, double grad_accuracy = kInfinity)
      : rbf_evaluator(
            model, source_points, source_grad_points,
            Bbox::from_points(source_points).convex_hull(Bbox::from_points(source_grad_points)),
            accuracy, grad_accuracy) {}

  rbf_evaluator(const Model& model, const Points& source_points, const Bbox& bbox,
                double accuracy = kInfinity)
      : rbf_evaluator(model, source_points, Points(0, kDim), bbox, accuracy, kInfinity) {}

  rbf_evaluator(const Model& model, const Points& source_points, const Points& source_grad_points,
                const Bbox& bbox, double accuracy = kInfinity, double grad_accuracy = kInfinity)
      : rbf_evaluator(model, bbox, accuracy, grad_accuracy) {
    set_source_points(source_points, source_grad_points);
  }

  rbf_evaluator(const Model& model, const Bbox& bbox, double accuracy = kInfinity,
                double grad_accuracy = kInfinity)
      : l_(model.poly_basis_size()), accuracy_(accuracy), grad_accuracy_(grad_accuracy) {
    for (const auto& rbf : model.rbfs()) {
      a_.push_back(fmm::make_fmm_evaluator(rbf, bbox));
      f_.push_back(fmm::make_fmm_gradient_evaluator(rbf, bbox));
      ft_.push_back(fmm::make_fmm_gradient_transpose_evaluator(rbf, bbox));
      h_.push_back(fmm::make_fmm_hessian_evaluator(rbf, bbox));
    }

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  vectord evaluate() const {
    vectord y = vectord::Zero(trg_mu_ + kDim * trg_sigma_);

    for (std::size_t i = 0; i < a_.size(); ++i) {
      y.head(trg_mu_) += a_.at(i)->evaluate();
      y.head(trg_mu_) += f_.at(i)->evaluate();
      y.tail(kDim * trg_sigma_) += ft_.at(i)->evaluate();
      y.tail(kDim * trg_sigma_) += h_.at(i)->evaluate();
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  vectord evaluate(const Points& target_points) { return evaluate(target_points, Points(0, kDim)); }

  vectord evaluate(const Points& target_points, const Points& target_grad_points) {
    set_target_points(target_points, target_grad_points);

    return evaluate();
  }

  void set_source_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    auto accuracy = (sigma_ > 0 ? accuracy_ / 2.0 : accuracy_) / static_cast<double>(a_.size());
    auto grad_accuracy =
        (sigma_ > 0 ? grad_accuracy_ / 2.0 : grad_accuracy_) / static_cast<double>(a_.size());

    for (std::size_t i = 0; i < a_.size(); ++i) {
      a_.at(i)->set_source_points(points);
      f_.at(i)->set_source_points(grad_points);
      ft_.at(i)->set_source_points(points);
      h_.at(i)->set_source_points(grad_points);

      a_.at(i)->set_accuracy(accuracy);
      f_.at(i)->set_accuracy(accuracy);
      ft_.at(i)->set_accuracy(grad_accuracy);
      h_.at(i)->set_accuracy(grad_accuracy);
    }
  }

  void set_target_points(const Points& points) { set_target_points(points, Points(0, kDim)); }

  void set_target_points(const Points& points, const Points& grad_points) {
    trg_mu_ = points.rows();
    trg_sigma_ = grad_points.rows();

    for (std::size_t i = 0; i < a_.size(); ++i) {
      a_.at(i)->set_target_points(points);
      f_.at(i)->set_target_points(points);
      ft_.at(i)->set_target_points(grad_points);
      h_.at(i)->set_target_points(grad_points);
    }

    if (l_ > 0) {
      p_->set_target_points(points, grad_points);
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    for (std::size_t i = 0; i < a_.size(); ++i) {
      a_.at(i)->set_weights(weights.head(mu_));
      f_.at(i)->set_weights(weights.segment(mu_, kDim * sigma_));
      ft_.at(i)->set_weights(weights.head(mu_));
      h_.at(i)->set_weights(weights.segment(mu_, kDim * sigma_));
    }

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const index_t l_;
  const double accuracy_;
  const double grad_accuracy_;
  index_t mu_{};
  index_t sigma_{};
  index_t trg_mu_{};
  index_t trg_sigma_{};

  std::vector<FmmGenericEvaluatorPtr> a_;
  std::vector<FmmGenericEvaluatorPtr> f_;
  std::vector<FmmGenericEvaluatorPtr> ft_;
  std::vector<FmmGenericEvaluatorPtr> h_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
