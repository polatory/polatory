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
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_direct_evaluator(const Model& model, const geometry::points3d& source_points,
                       const geometry::points3d& source_grad_points)
      : model_(model),
        dim_(model.poly_dimension()),
        l_(model.poly_basis_size()),
        mu_(source_points.rows()),
        sigma_(source_grad_points.rows()),
        src_points_(source_points),
        src_grad_points_(source_grad_points) {
    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
    }
  }

  common::valuesd evaluate() const {
    const auto& rbf = model_.rbf();
    auto w = weights_.head(mu_);
    auto grad_w = weights_.segment(mu_, dim_ * sigma_).reshaped<Eigen::RowMajor>(sigma_, dim_);

    common::valuesd y = common::valuesd::Zero(fld_mu_ + dim_ * fld_sigma_);

    for (index_t i = 0; i < fld_mu_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        y(i) += w(j) * rbf.evaluate(fld_points_.row(i) - src_points_.row(j));
      }

      for (index_t j = 0; j < sigma_; j++) {
        y(i) += grad_w.row(j).dot(
            -rbf.evaluate_gradient(fld_points_.row(i) - src_grad_points_.row(j)).head(dim_));
      }
    }

    for (index_t i = 0; i < fld_sigma_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        y.segment(fld_mu_ + dim_ * i, dim_) +=
            w(j) * rbf.evaluate_gradient(fld_grad_points_.row(i) - src_points_.row(j))
                       .head(dim_)
                       .transpose();
      }

      for (index_t j = 0; j < sigma_; j++) {
        y.segment(fld_mu_ + dim_ * i, dim_) +=
            (grad_w.row(j) * rbf.evaluate_hessian(fld_grad_points_.row(i) - src_grad_points_.row(j))
                                 .topLeftCorner(dim_, dim_))
                .transpose();
      }
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  void set_field_points(const geometry::points3d& field_points,
                        const geometry::points3d& field_grad_points) {
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
    POLATORY_ASSERT(weights.rows() == mu_ + dim_ * sigma_ + l_);

    weights_ = weights;

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const Model& model_;
  const int dim_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const geometry::points3d src_points_;
  const geometry::points3d src_grad_points_;

  std::unique_ptr<PolynomialEvaluator> p_;

  index_t fld_mu_{};
  index_t fld_sigma_{};
  geometry::points3d fld_points_;
  geometry::points3d fld_grad_points_;
  common::valuesd weights_;
};

}  // namespace polatory::interpolation
