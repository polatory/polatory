#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/krylov/linear_operator.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <int Dim>
class DirectOperator : public krylov::LinearOperator {
  static constexpr int kDim = Dim;
  using Model = Model<kDim>;
  using MonomialBasis = polynomial::MonomialBasis<kDim>;
  using Points = geometry::Points<kDim>;
  using Vector = geometry::Vector<kDim>;

 public:
  DirectOperator(const Model& model, const Points& points, const Points& grad_points)
      : DirectOperator(model) {
    set_points(points, grad_points);
  }

  explicit DirectOperator(const Model& model) : model_(model), l_(model.poly_basis_size()) {}

  VecX operator()(const VecX& weights) const override {
    POLATORY_ASSERT(weights.rows() == size());

    auto w = weights.head(mu_);
    auto grad_w = weights.segment(mu_, kDim * sigma_).reshaped<Eigen::RowMajor>(sigma_, kDim);

    VecX y = VecX::Zero(size());

    y.head(mu_) = weights.head(mu_) * model_.nugget();

    for (const auto& rbf : model_.rbfs()) {
#pragma omp parallel
      {
        VecX y_local = VecX::Zero(size());

#pragma omp for
        for (Index i = 0; i < mu_; i++) {
          for (Index j = 0; j < mu_; j++) {
            Vector diff = points_.row(i) - points_.row(j);
            y_local(i) += w(j) * rbf.evaluate(diff);
          }

          for (Index j = 0; j < sigma_; j++) {
            Vector diff = points_.row(i) - grad_points_.row(j);
            y_local(i) += grad_w.row(j).dot(-rbf.evaluate_gradient(diff));
          }
        }

#pragma omp for
        for (Index i = 0; i < sigma_; i++) {
          for (Index j = 0; j < mu_; j++) {
            Vector diff = grad_points_.row(i) - points_.row(j);
            y_local.segment<kDim>(mu_ + kDim * i) += w(j) * rbf.evaluate_gradient(diff).transpose();
          }

          for (Index j = 0; j < sigma_; j++) {
            Vector diff = grad_points_.row(i) - grad_points_.row(j);
            y_local.segment<kDim>(mu_ + kDim * i) +=
                (grad_w.row(j) * -rbf.evaluate_hessian(diff)).transpose();
          }
        }

#pragma omp critical
        y += y_local;
      }
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
    points_ = points;
    grad_points_ = grad_points;

    if (l_ > 0) {
      MonomialBasis basis(model_.poly_degree());
      p_ = basis.evaluate(points_, grad_points_);
    }
  }

  Index size() const override { return mu_ + kDim * sigma_ + l_; }

 private:
  const Model& model_;
  const Index l_;
  Index mu_{};
  Index sigma_{};
  Points points_;
  Points grad_points_;

  MatX p_;
};

}  // namespace polatory::interpolation
