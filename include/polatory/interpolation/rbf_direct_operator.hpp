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

template <int Dim>
class rbf_direct_operator : public krylov::linear_operator {
  static constexpr int kDim = Dim;
  using Model = model<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Vector = geometry::vectorNd<kDim>;

 public:
  rbf_direct_operator(const Model& model, const Points& points, const Points& grad_points)
      : rbf_direct_operator(model) {
    set_points(points, grad_points);
  }

  rbf_direct_operator(const Model& model) : model_(model), l_(model.poly_basis_size()) {}

  common::valuesd operator()(const common::valuesd& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    auto w = weights.head(mu_);
    auto grad_w = weights.segment(mu_, kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim);

    common::valuesd y = common::valuesd::Zero(size());

    for (const auto& rbf : model_.rbfs()) {
      for (index_t i = 0; i < mu_; i++) {
        for (index_t j = 0; j < mu_; j++) {
          Vector diff = points_.row(i) - points_.row(j);
          y(i) += w(j) * rbf->evaluate(diff);
        }

        for (index_t j = 0; j < sigma_; j++) {
          Vector diff = points_.row(i) - grad_points_.row(j);
          y(i) += grad_w.row(j).dot(-rbf->evaluate_gradient(diff));
        }
      }

      for (index_t i = 0; i < sigma_; i++) {
        for (index_t j = 0; j < mu_; j++) {
          Vector diff = grad_points_.row(i) - points_.row(j);
          y.segment<kDim>(mu_ + kDim * i) += w(j) * rbf->evaluate_gradient(diff).transpose();
        }

        for (index_t j = 0; j < sigma_; j++) {
          Vector diff = grad_points_.row(i) - grad_points_.row(j);
          y.segment<kDim>(mu_ + kDim * i) +=
              (grad_w.row(j) * -rbf->evaluate_hessian(diff)).transpose();
        }
      }
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
    points_ = points;
    grad_points_ = grad_points;

    if (l_ > 0) {
      MonomialBasis basis(model_.poly_degree());
      pt_ = basis.evaluate(points_, grad_points_);
    }
  }

  index_t size() const override { return mu_ + kDim * sigma_ + l_; }

 private:
  const Model& model_;
  const index_t l_;
  index_t mu_;
  index_t sigma_;
  Points points_;
  Points grad_points_;

  Eigen::MatrixXd pt_;
};

}  // namespace polatory::interpolation
