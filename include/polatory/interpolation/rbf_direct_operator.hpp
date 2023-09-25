#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/numeric/sum_accumulator.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

class rbf_direct_operator : krylov::linear_operator {
  using PolynomialEvaluator = polynomial::polynomial_evaluator<polynomial::monomial_basis>;

 public:
  rbf_direct_operator(const model& model, const geometry::points3d& points)
      : rbf_direct_operator(model, points, geometry::points3d(0, 3)) {}

  rbf_direct_operator(const model& model, const geometry::points3d& points,
                      const geometry::points3d& grad_points)
      : model_(model),
        dim_(model.poly_dimension()),
        l_(model.poly_basis_size()),
        mu_(points.rows()),
        sigma_(grad_points.rows()),
        points_(points),
        grad_points_(grad_points) {
    if (l_ > 0) {
      polynomial::monomial_basis basis(model_.poly_dimension(), model_.poly_degree());
      pt_ = basis.evaluate(points_, grad_points_);
    }
  }

  // TODO: Use Kahan summation.
  common::valuesd operator()(const common::valuesd& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    common::valuesd y = common::valuesd::Zero(size());

    auto w = weights.head(mu_);
    auto grad_w = weights.segment(mu_, dim_ * sigma_).reshaped<Eigen::RowMajor>(sigma_, dim_);

    const auto& rbf = model_.rbf();

    for (index_t i = 0; i < mu_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        y(i) += w(j) * rbf.evaluate(points_.row(i) - points_.row(j));
      }

      for (index_t j = 0; j < sigma_; j++) {
        y(i) += grad_w.row(j).dot(
            -rbf.evaluate_gradient(points_.row(i) - grad_points_.row(j)).head(dim_));
      }
    }

    for (index_t i = 0; i < sigma_; i++) {
      for (index_t j = 0; j < mu_; j++) {
        y.segment(mu_ + dim_ * i, dim_) +=
            w(j) *
            -rbf.evaluate_gradient(grad_points_.row(i) - points_.row(j)).head(dim_).transpose();
      }

      for (index_t j = 0; j < sigma_; j++) {
        y.segment(mu_ + dim_ * i, dim_) +=
            (grad_w.row(j) * rbf.evaluate_hessian(grad_points_.row(i) - grad_points_.row(j))
                                 .topLeftCorner(dim_, dim_))
                .transpose();
      }
    }

    if (l_ > 0) {
      // Add polynomial terms.
      y.head(mu_ + dim_ * sigma_) += pt_.transpose() * weights.tail(l_);
      y.tail(l_) += pt_ * weights.head(mu_ + dim_ * sigma_);
    }

    y.head(mu_) += weights.head(mu_) * model_.nugget();

    return y;
  }

  index_t size() const override { return mu_ + dim_ * sigma_ + l_; }

 private:
  const model& model_;
  const int dim_;
  const index_t l_;
  const index_t mu_;
  const index_t sigma_;
  const geometry::points3d points_;
  const geometry::points3d grad_points_;

  Eigen::MatrixXd pt_;
};

}  // namespace polatory::interpolation
