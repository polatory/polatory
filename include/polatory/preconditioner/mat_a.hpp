#pragma once

#include <Eigen/Core>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::preconditioner {

template <int Dim, class DerivedPoints, class DerivedGradPoints>
Eigen::MatrixXd mat_a(const model<Dim>& model, const Eigen::MatrixBase<DerivedPoints>& points,
                      const Eigen::MatrixBase<DerivedGradPoints>& grad_points) {
  constexpr int kDim = Dim;
  using Vector = geometry::vectorNd<kDim>;

  const auto& rbf = model.rbf();
  auto mu = points.rows();
  auto sigma = grad_points.rows();
  auto m = mu + kDim * sigma;

  Eigen::MatrixXd a(m, m);

  auto aa = a.topLeftCorner(mu, mu);
  aa.diagonal().array() = rbf->evaluate(Vector::Zero()) + model.nugget();
  for (index_t i = 0; i < mu - 1; i++) {
    for (index_t j = i + 1; j < mu; j++) {
      Vector diff = points.row(i) - points.row(j);
      aa(i, j) = rbf->evaluate(diff);
      aa(j, i) = aa(i, j);
    }
  }

  if (sigma > 0) {
    auto af = a.topRightCorner(mu, kDim * sigma);
    for (index_t i = 0; i < mu; i++) {
      for (index_t j = 0; j < sigma; j++) {
        Vector diff = points.row(i) - grad_points.row(j);
        af.template block<1, kDim>(i, kDim * j) = -rbf->evaluate_gradient(diff);
      }
    }
    a.bottomLeftCorner(kDim * sigma, mu) = af.transpose();

    auto ah = a.bottomRightCorner(kDim * sigma, kDim * sigma);
    Eigen::MatrixXd ah_diagonal = -rbf->evaluate_hessian(Vector::Zero());
    for (index_t i = 0; i < sigma; i++) {
      ah.template block<kDim, kDim>(kDim * i, kDim * i) = ah_diagonal;
    }
    for (index_t i = 0; i < sigma - 1; i++) {
      for (index_t j = i + 1; j < sigma; j++) {
        Vector diff = grad_points.row(i) - grad_points.row(j);
        ah.template block<kDim, kDim>(kDim * i, kDim * j) = -rbf->evaluate_hessian(diff);
        ah.template block<kDim, kDim>(kDim * j, kDim * i) =
            ah.template block<kDim, kDim>(kDim * i, kDim * j).transpose();
      }
    }
  }

  return a;
}

}  // namespace polatory::preconditioner
