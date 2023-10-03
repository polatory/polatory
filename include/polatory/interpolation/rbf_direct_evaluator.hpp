#pragma once

#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <class Model>
class rbf_direct_evaluator {
  static constexpr int kDim = Model::kDim;
  using Points = geometry::pointsNd<kDim>;
  using Vector = geometry::vectorNd<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using PolynomialEvaluator = polynomial::polynomial_evaluator<MonomialBasis>;

 public:
  rbf_direct_evaluator(const Model& model, const Points& source_points,
                       const Points& source_grad_points)
      : rbf_direct_evaluator(model) {
    set_source_points(source_points, source_grad_points);
  }

  rbf_direct_evaluator(const Model& model) : model_(model), l_(model.poly_basis_size()) {
    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  common::valuesd evaluate() const {
    const auto& rbf = model_.rbf();
    auto w = weights_.head(mu_);
    auto grad_w = weights_.segment(mu_, kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim);

    common::valuesd y = common::valuesd::Zero(trg_mu_ + kDim * trg_sigma_);

    for (index_t i = 0; i < trg_mu_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        Vector diff = trg_points_.row(i) - src_points_.row(j);
        y(i) += w(j) * rbf.evaluate(diff);
      }

      for (index_t j = 0; j < sigma_; j++) {
        Vector diff = trg_points_.row(i) - src_grad_points_.row(j);
        y(i) += grad_w.row(j).dot(-rbf.evaluate_gradient(diff));
      }
    }

    for (index_t i = 0; i < trg_sigma_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        Vector diff = trg_grad_points_.row(i) - src_points_.row(j);
        y.segment(trg_mu_ + kDim * i, kDim) += w(j) * rbf.evaluate_gradient(diff).transpose();
      }

      for (index_t j = 0; j < sigma_; j++) {
        Vector diff = trg_grad_points_.row(i) - src_grad_points_.row(j);
        y.segment(trg_mu_ + kDim * i, kDim) +=
            (grad_w.row(j) * -rbf.evaluate_hessian(diff)).transpose();
      }
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  void set_source_points(const Points& source_points, const Points& source_grad_points) {
    mu_ = static_cast<index_t>(source_points.rows());
    sigma_ = static_cast<index_t>(source_grad_points.rows());

    src_points_ = source_points;
    src_grad_points_ = source_grad_points;
  }

  void set_target_points(const Points& target_points) {
    set_target_points(target_points, Points(0, kDim));
  }

  void set_target_points(const Points& target_points, const Points& target_grad_points) {
    trg_mu_ = static_cast<index_t>(target_points.rows());
    trg_sigma_ = static_cast<index_t>(target_grad_points.rows());

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
  const index_t l_;
  std::unique_ptr<PolynomialEvaluator> p_;

  index_t mu_;
  index_t sigma_;
  index_t trg_mu_{};
  index_t trg_sigma_{};
  Points src_points_;
  Points src_grad_points_;
  Points trg_points_;
  Points trg_grad_points_;
  common::valuesd weights_;
};

}  // namespace polatory::interpolation
