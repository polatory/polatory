#pragma once

#include <Eigen/Cholesky>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::kriging {

template <int Dim>
vectord detrend(const geometry::pointsNd<Dim>& points, const vectord& values, int degree) {
  if (degree < 0 || degree > 2) {
    throw std::invalid_argument("degree must be 0, 1, or 2");
  }

  polynomial::monomial_basis<Dim> basis(degree);

  auto p = basis.evaluate(points);

  matrixd system = p.transpose() * p;
  vectord rhs = p.transpose() * values;

  vectord coeffs = system.ldlt().solve(rhs);

  return values - p * coeffs;
}

}  // namespace polatory::kriging
