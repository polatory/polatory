#pragma once

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/fmm/resource.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::interpolation {

template <int Dim>
class Evaluator {
  static constexpr int kDim = Dim;
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Bbox = geometry::Bbox<kDim>;
  using GradResource = fmm::Resource<kDim, kDim>;
  using Model = Model<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Points = geometry::Points<kDim>;
  using PolynomialEvaluator = polynomial::PolynomialEvaluator<MonomialBasis>;
  using Resource = fmm::Resource<kDim, 1>;

  template <int km, int kn>
  using FmmGenericEvaluatorPtr = fmm::FmmGenericEvaluatorPtr<kDim, km, kn>;

 public:
  Evaluator(const Model& model, const Points& source_points, double accuracy = kInfinity)
      : Evaluator(model, source_points, Points(0, kDim), accuracy, kInfinity) {}

  Evaluator(const Model& model, const Points& source_points, const Points& source_grad_points,
            double accuracy = kInfinity, double grad_accuracy = kInfinity)
      : Evaluator(
            model, source_points, source_grad_points,
            Bbox::from_points(source_points).convex_hull(Bbox::from_points(source_grad_points)),
            accuracy, grad_accuracy) {}

  Evaluator(const Model& model, const Points& source_points, const Bbox& bbox,
            double accuracy = kInfinity)
      : Evaluator(model, source_points, Points(0, kDim), bbox, accuracy, kInfinity) {}

  Evaluator(const Model& model, const Points& source_points, const Points& source_grad_points,
            const Bbox& bbox, double accuracy = kInfinity, double grad_accuracy = kInfinity)
      : Evaluator(model, bbox, accuracy, grad_accuracy) {
    set_source_points(source_points, source_grad_points);
  }

  Evaluator(const Model& model, const Bbox& bbox, double accuracy = kInfinity,
            double grad_accuracy = kInfinity)
      : l_(model.poly_basis_size()), accuracy_(accuracy), grad_accuracy_(grad_accuracy) {
    for (const auto& rbf : model.rbfs()) {
      src_resources_.emplace_back(rbf, bbox);
      src_grad_resources_.emplace_back(rbf, bbox);
      trg_resources_.emplace_back(rbf, bbox);
      trg_grad_resources_.emplace_back(rbf, bbox);

      a_.push_back(fmm::make_fmm_evaluator(rbf, bbox));
      f_.push_back(fmm::make_fmm_gradient_evaluator(rbf, bbox));
      ft_.push_back(fmm::make_fmm_gradient_transpose_evaluator(rbf, bbox));
      h_.push_back(fmm::make_fmm_hessian_evaluator(rbf, bbox));
    }

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  VecX evaluate() const {
    VecX y = VecX::Zero(trg_mu_ + kDim * trg_sigma_);

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

  VecX evaluate(const Points& target_points) { return evaluate(target_points, Points(0, kDim)); }

  VecX evaluate(const Points& target_points, const Points& target_grad_points) {
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
      src_resources_.at(i).set_points(points);
      src_grad_resources_.at(i).set_points(grad_points);

      a_.at(i)->set_source_resource(src_resources_.at(i));
      f_.at(i)->set_source_resource(src_grad_resources_.at(i));
      ft_.at(i)->set_source_resource(src_resources_.at(i));
      h_.at(i)->set_source_resource(src_grad_resources_.at(i));

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
      trg_resources_.at(i).set_points(points);
      trg_grad_resources_.at(i).set_points(grad_points);

      a_.at(i)->set_target_resource(trg_resources_.at(i));
      f_.at(i)->set_target_resource(trg_resources_.at(i));
      ft_.at(i)->set_target_resource(trg_grad_resources_.at(i));
      h_.at(i)->set_target_resource(trg_grad_resources_.at(i));
    }

    if (l_ > 0) {
      p_->set_target_points(points, grad_points);
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    for (std::size_t i = 0; i < a_.size(); ++i) {
      src_resources_.at(i).set_weights(weights.head(mu_));
      src_grad_resources_.at(i).set_weights(weights.segment(mu_, kDim * sigma_));
    }

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const Index l_;
  const double accuracy_;
  const double grad_accuracy_;
  Index mu_{};
  Index sigma_{};
  Index trg_mu_{};
  Index trg_sigma_{};

  std::vector<Resource> src_resources_;
  std::vector<GradResource> src_grad_resources_;
  std::vector<Resource> trg_resources_;
  std::vector<GradResource> trg_grad_resources_;
  std::vector<FmmGenericEvaluatorPtr<1, 1>> a_;
  std::vector<FmmGenericEvaluatorPtr<kDim, 1>> f_;
  std::vector<FmmGenericEvaluatorPtr<1, kDim>> ft_;
  std::vector<FmmGenericEvaluatorPtr<kDim, kDim>> h_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
