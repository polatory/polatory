#pragma once

#include <Eigen/Core>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::preconditioner {

template <class DerivedPoints, class DerivedGradPoints>
Eigen::MatrixXd mat_a(const model& model, const Eigen::MatrixBase<DerivedPoints>& points,
                      const Eigen::MatrixBase<DerivedGradPoints>& grad_points) {
  const auto& rbf = model.rbf();
  auto dim = model.poly_dimension();
  auto mu = points.rows();
  auto sigma = grad_points.rows();
  auto m = mu + dim * sigma;

  Eigen::MatrixXd a(m, m);

  auto aa = a.topLeftCorner(mu, mu);
  aa.diagonal().array() = rbf.evaluate(geometry::vector3d::Zero()) + model.nugget();
  for (index_t i = 0; i < mu - 1; i++) {
    for (index_t j = i + 1; j < mu; j++) {
      aa(i, j) = rbf.evaluate(points.row(i) - points.row(j));
      aa(j, i) = aa(i, j);
    }
  }

  if (sigma > 0) {
    auto af = a.block(0, mu, mu, dim * sigma);
    for (index_t i = 0; i < mu; i++) {
      for (index_t j = 0; j < sigma; j++) {
        af.block(i, dim * j, 1, dim) = -rbf.evaluate_gradient(points.row(i) - grad_points.row(j));
      }
    }
    a.block(mu, 0, dim * sigma, mu) = af.transpose();

    auto ah = a.block(mu, mu, dim * sigma, dim * sigma);
    auto ah_diagonal = rbf.evaluate_hessian(geometry::vector3d::Zero());
    for (index_t i = 0; i < sigma; i++) {
      ah.block(dim * i, dim * i, dim, dim) = ah_diagonal;
    }
    for (index_t i = 0; i < sigma - 1; i++) {
      for (index_t j = i + 1; j < sigma; j++) {
        ah.block(dim * i, dim * j, dim, dim) =
            rbf.evaluate_hessian(grad_points.row(i) - grad_points.row(j));
      }
    }
  }

  return a;
}

}  // namespace polatory::preconditioner
