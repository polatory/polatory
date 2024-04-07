#pragma once

#include <Eigen/Core>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::preconditioner {

template <int Dim, class DerivedPoints, class DerivedGradPoints>
matrixd mat_a(const model<Dim>& model, const Eigen::MatrixBase<DerivedPoints>& points,
              const Eigen::MatrixBase<DerivedGradPoints>& grad_points) {
  constexpr int kDim = Dim;
  using Vector = geometry::vectorNd<kDim>;

  auto mu = points.rows();
  auto sigma = grad_points.rows();
  auto m = mu + kDim * sigma;

  matrixd a = matrixd::Zero(m, m);

  auto aa = a.topLeftCorner(mu, mu);
  aa.diagonal().array() = model.nugget();

  for (const auto& rbf : model.rbfs()) {
    aa.diagonal().array() += rbf.evaluate(Vector::Zero());
    for (index_t i = 0; i < mu - 1; i++) {
      for (index_t j = i + 1; j < mu; j++) {
        Vector diff = points.row(i) - points.row(j);
        aa(i, j) += rbf.evaluate(diff);
      }
    }

    if (sigma > 0) {
      auto af = a.topRightCorner(mu, kDim * sigma);
      for (index_t i = 0; i < mu; i++) {
        for (index_t j = 0; j < sigma; j++) {
          Vector diff = points.row(i) - grad_points.row(j);
          af.template block<1, kDim>(i, kDim * j) += -rbf.evaluate_gradient(diff);
        }
      }

      auto ah = a.bottomRightCorner(kDim * sigma, kDim * sigma);
      matrixd ah_diagonal = -rbf.evaluate_hessian(Vector::Zero());
      for (index_t i = 0; i < sigma; i++) {
        ah.template block<kDim, kDim>(kDim * i, kDim * i) += ah_diagonal;
      }
      for (index_t i = 0; i < sigma - 1; i++) {
        for (index_t j = i + 1; j < sigma; j++) {
          Vector diff = grad_points.row(i) - grad_points.row(j);
          ah.template block<kDim, kDim>(kDim * i, kDim * j) += -rbf.evaluate_hessian(diff);
        }
      }
    }
  }

  a.triangularView<Eigen::Lower>() = a.transpose().triangularView<Eigen::Lower>();

  return a;
}

}  // namespace polatory::preconditioner
