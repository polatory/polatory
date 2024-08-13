#pragma once

#include <Eigen/Cholesky>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::kriging {

template <int Dim>
VecX detrend(const geometry::Points<Dim>& points, const VecX& values, int degree) {
  if (degree < 0 || degree > 2) {
    throw std::invalid_argument("degree must be 0, 1, or 2");
  }

  polynomial::MonomialBasis<Dim> basis(degree);

  auto p = basis.evaluate(points);

  MatX system = p.transpose() * p;
  VecX rhs = p.transpose() * values;

  VecX coeffs = system.ldlt().solve(rhs);

  return values - p * coeffs;
}

}  // namespace polatory::kriging
