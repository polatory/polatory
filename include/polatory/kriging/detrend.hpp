#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::kriging {

template <int Dim>
common::valuesd detrend(const geometry::pointsNd<Dim>& points,
                        const geometry::pointsNd<Dim>& grad_points, const common::valuesd& values,
                        int degree) {
  if (degree < 0 || degree > 2) {
    throw std::invalid_argument("degree must be 0, 1, or 2.");
  }

  polynomial::monomial_basis<Dim> basis(degree);

  Eigen::MatrixXd pt = basis.evaluate(points, grad_points);

  Eigen::MatrixXd system = pt * pt.transpose();
  Eigen::VectorXd rhs = pt * values;

  Eigen::VectorXd coeffs = system.ldlt().solve(rhs);

  return values - pt.transpose() * coeffs;
}

template <int Dim>
common::valuesd detrend(const geometry::pointsNd<Dim>& points, const common::valuesd& values,
                        int degree) {
  return detrend(points, geometry::pointsNd<Dim>(0, Dim), values, degree);
}

}  // namespace polatory::kriging
