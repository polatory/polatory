#pragma once

#include <Eigen/Core>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::preconditioner {

template <class Model, class DerivedPoints, class DerivedGradPoints>
Eigen::MatrixXd mat_a(const Model& model, const Eigen::MatrixBase<DerivedPoints>& points,
                      const Eigen::MatrixBase<DerivedGradPoints>& grad_points) {
  constexpr int kDim = Model::kDim;
  using Vector = geometry::vectorNd<kDim>;

  const auto& rbf = model.rbf();
  auto mu = points.rows();
  auto sigma = grad_points.rows();
  auto m = mu + kDim * sigma;

  Eigen::MatrixXd a(m, m);

  auto aa = a.topLeftCorner(mu, mu);
  aa.diagonal().array() = rbf.evaluate(Vector::Zero()) + model.nugget();
  for (index_t i = 0; i < mu - 1; i++) {
    for (index_t j = i + 1; j < mu; j++) {
      aa(i, j) = rbf.evaluate(points.row(i) - points.row(j));
      aa(j, i) = aa(i, j);
    }
  }

  if (sigma > 0) {
    auto af = a.topRightCorner(mu, kDim * sigma);
    for (index_t i = 0; i < mu; i++) {
      for (index_t j = 0; j < sigma; j++) {
        af.block(i, kDim * j, 1, kDim) =
            -rbf.evaluate_gradient(points.row(i) - grad_points.row(j)).head(kDim);
      }
    }
    a.bottomLeftCorner(kDim * sigma, mu) = af.transpose();

    auto ah = a.bottomRightCorner(kDim * sigma, kDim * sigma);
    Eigen::MatrixXd ah_diagonal = rbf.evaluate_hessian(Vector::Zero()).topLeftCorner(kDim, kDim);
    for (index_t i = 0; i < sigma; i++) {
      ah.block(kDim * i, kDim * i, kDim, kDim) = ah_diagonal;
    }
    for (index_t i = 0; i < sigma - 1; i++) {
      for (index_t j = i + 1; j < sigma; j++) {
        ah.block(kDim * i, kDim * j, kDim, kDim) =
            rbf.evaluate_hessian(grad_points.row(i) - grad_points.row(j)).topLeftCorner(kDim, kDim);
        ah.block(kDim * j, kDim * i, kDim, kDim) =
            ah.block(kDim * i, kDim * j, kDim, kDim).transpose();
      }
    }
  }

  return a;
}

}  // namespace polatory::preconditioner
