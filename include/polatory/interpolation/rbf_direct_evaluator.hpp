#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Dim>
class DirectEvaluator {
  static constexpr int kDim = Dim;
  using Model = Model<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Points = geometry::Points<kDim>;
  using PolynomialEvaluator = polynomial::PolynomialEvaluator<MonomialBasis>;
  using Vector = geometry::Vector<kDim>;

 public:
  DirectEvaluator(const Model& model, const Points& source_points)
      : DirectEvaluator(model, source_points, Points(0, kDim)) {}

  DirectEvaluator(const Model& model, const Points& source_points, const Points& source_grad_points)
      : DirectEvaluator(model) {
    set_source_points(source_points, source_grad_points);
  }

  explicit DirectEvaluator(const Model& model) : model_(model), l_(model.poly_basis_size()) {
    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  VecX evaluate() const {
    auto w = weights_.head(mu_);
    auto grad_w = weights_.segment(mu_, kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim);

    VecX y = VecX::Zero(trg_mu_ + kDim * trg_sigma_);

    for (const auto& rbf : model_.rbfs()) {
#pragma omp parallel
      {
        VecX y_local = VecX::Zero(trg_mu_ + kDim * trg_sigma_);

#pragma omp for
        for (Index i = 0; i < trg_mu_; i++) {
          for (Index j = 0; j < mu_; j++) {
            Vector diff = trg_points_.row(i) - src_points_.row(j);
            y_local(i) += w(j) * rbf.evaluate(diff);
          }

          for (Index j = 0; j < sigma_; j++) {
            Vector diff = trg_points_.row(i) - src_grad_points_.row(j);
            y_local(i) += grad_w.row(j).dot(-rbf.evaluate_gradient(diff));
          }
        }

#pragma omp for
        for (Index i = 0; i < trg_sigma_; i++) {
          for (Index j = 0; j < mu_; j++) {
            Vector diff = trg_grad_points_.row(i) - src_points_.row(j);
            y_local.segment<kDim>(trg_mu_ + kDim * i) +=
                w(j) * rbf.evaluate_gradient(diff).transpose();
          }

          for (Index j = 0; j < sigma_; j++) {
            Vector diff = trg_grad_points_.row(i) - src_grad_points_.row(j);
            y_local.segment<kDim>(trg_mu_ + kDim * i) +=
                (grad_w.row(j) * -rbf.evaluate_hessian(diff)).transpose();
          }
        }

#pragma omp critical
        y += y_local;
      }
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

  void set_source_points(const Points& source_points, const Points& source_grad_points) {
    mu_ = static_cast<Index>(source_points.rows());
    sigma_ = static_cast<Index>(source_grad_points.rows());

    src_points_ = source_points;
    src_grad_points_ = source_grad_points;
  }

  void set_target_points(const Points& target_points) {
    set_target_points(target_points, Points(0, kDim));
  }

  void set_target_points(const Points& target_points, const Points& target_grad_points) {
    trg_mu_ = static_cast<Index>(target_points.rows());
    trg_sigma_ = static_cast<Index>(target_grad_points.rows());

    trg_points_ = target_points;
    trg_grad_points_ = target_grad_points;

    if (l_ > 0) {
      p_->set_target_points(trg_points_, trg_grad_points_);
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    weights_ = weights;

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const Model& model_;
  const Index l_;
  std::unique_ptr<PolynomialEvaluator> p_;

  Index mu_;
  Index sigma_;
  Index trg_mu_{};
  Index trg_sigma_{};
  Points src_points_;
  Points src_grad_points_;
  Points trg_points_;
  Points trg_grad_points_;
  VecX weights_;
};

}  // namespace polatory::interpolation
