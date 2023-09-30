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
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using PolynomialEvaluator = polynomial::polynomial_evaluator<MonomialBasis>;

 public:
  rbf_direct_evaluator(const Model& model, const Points& source_points,
                       const Points& source_grad_points)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_(source_points.rows()),
        sigma_(source_grad_points.rows()),
        src_points_(source_points),
        src_grad_points_(source_grad_points) {
    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  common::valuesd evaluate() const {
    const auto& rbf = model_.rbf();
    auto w = weights_.head(mu_);
    auto grad_w = weights_.segment(mu_, kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim);

    common::valuesd y = common::valuesd::Zero(fld_mu_ + kDim * fld_sigma_);

    for (index_t i = 0; i < fld_mu_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        y(i) += w(j) * rbf.evaluate(fld_points_.row(i) - src_points_.row(j));
      }

      for (index_t j = 0; j < sigma_; j++) {
        y(i) += grad_w.row(j).dot(
            -rbf.evaluate_gradient(fld_points_.row(i) - src_grad_points_.row(j)).head(kDim));
      }
    }

    for (index_t i = 0; i < fld_sigma_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        y.segment(fld_mu_ + kDim * i, kDim) +=
            w(j) * rbf.evaluate_gradient(fld_grad_points_.row(i) - src_points_.row(j))
                       .head(kDim)
                       .transpose();
      }

      for (index_t j = 0; j < sigma_; j++) {
        y.segment(fld_mu_ + kDim * i, kDim) +=
            (grad_w.row(j) * rbf.evaluate_hessian(fld_grad_points_.row(i) - src_grad_points_.row(j))
                                 .topLeftCorner(kDim, kDim))
                .transpose();
      }
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  void set_field_points(const Points& field_points, const Points& field_grad_points) {
    fld_mu_ = static_cast<index_t>(field_points.rows());
    fld_sigma_ = static_cast<index_t>(field_grad_points.rows());

    fld_points_ = field_points;
    fld_grad_points_ = field_grad_points;

    if (l_ > 0) {
      p_->set_field_points(fld_points_, fld_grad_points_);
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
  const index_t mu_;
  const index_t sigma_;
  const Points src_points_;
  const Points src_grad_points_;

  std::unique_ptr<PolynomialEvaluator> p_;

  index_t fld_mu_{};
  index_t fld_sigma_{};
  Points fld_points_;
  Points fld_grad_points_;
  common::valuesd weights_;
};

}  // namespace polatory::interpolation
