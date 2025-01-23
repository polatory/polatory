#pragma once

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
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
class SymmetricEvaluator {
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

  template <int kn>
  using FmmGenericSymmetricEvaluatorPtr = fmm::FmmGenericSymmetricEvaluatorPtr<kDim, kn>;

 public:
  SymmetricEvaluator(const Model& model, const Points& points, const Points& grad_points,
                     double accuracy = kInfinity, double grad_accuracy = kInfinity)
      : SymmetricEvaluator(model,
                           Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points)),
                           accuracy, grad_accuracy) {
    set_points(points, grad_points);
  }

  SymmetricEvaluator(const Model& model, const Bbox& bbox, double accuracy = kInfinity,
                     double grad_accuracy = kInfinity)
      : l_(model.poly_basis_size()), accuracy_(accuracy), grad_accuracy_(grad_accuracy) {
    for (const auto& rbf : model.rbfs()) {
      resources_.emplace_back(rbf, bbox);
      grad_resources_.emplace_back(rbf, bbox);

      a_.push_back(fmm::make_fmm_symmetric_evaluator(rbf, bbox));
      f_.push_back(fmm::make_fmm_gradient_evaluator(rbf, bbox));
      ft_.push_back(fmm::make_fmm_gradient_transpose_evaluator(rbf, bbox));
      h_.push_back(fmm::make_fmm_hessian_symmetric_evaluator(rbf, bbox));
    }

    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  VecX evaluate() const {
    VecX y = VecX::Zero(mu_ + kDim * sigma_);

    for (std::size_t i = 0; i < a_.size(); ++i) {
      y.head(mu_) += a_.at(i)->evaluate();
      y.head(mu_) += f_.at(i)->evaluate();
      y.tail(kDim * sigma_) += ft_.at(i)->evaluate();
      y.tail(kDim * sigma_) += h_.at(i)->evaluate();
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
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
      resources_.at(i).set_points(points);
      grad_resources_.at(i).set_points(grad_points);

      a_.at(i)->set_resource(resources_.at(i));
      f_.at(i)->set_source_resource(grad_resources_.at(i));
      f_.at(i)->set_target_resource(resources_.at(i));
      ft_.at(i)->set_source_resource(resources_.at(i));
      ft_.at(i)->set_target_resource(grad_resources_.at(i));
      h_.at(i)->set_resource(grad_resources_.at(i));

      a_.at(i)->set_accuracy(accuracy);
      f_.at(i)->set_accuracy(accuracy);
      ft_.at(i)->set_accuracy(grad_accuracy);
      h_.at(i)->set_accuracy(grad_accuracy);
    }

    if (l_ > 0) {
      p_->set_target_points(points, grad_points);
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    for (std::size_t i = 0; i < a_.size(); ++i) {
      resources_.at(i).set_weights(weights.head(mu_));
      grad_resources_.at(i).set_weights(weights.segment(mu_, kDim * sigma_));
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

  std::vector<Resource> resources_;
  std::vector<GradResource> grad_resources_;
  std::vector<FmmGenericSymmetricEvaluatorPtr<1>> a_;
  std::vector<FmmGenericEvaluatorPtr<kDim, 1>> f_;
  std::vector<FmmGenericEvaluatorPtr<1, kDim>> ft_;
  std::vector<FmmGenericSymmetricEvaluatorPtr<kDim>> h_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
