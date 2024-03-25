#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/types.hpp>

namespace polatory::kriging {

template <int Dim>
common::valuesd detrend(const geometry::pointsNd<Dim>& points,
                        const geometry::pointsNd<Dim>& grad_points, const common::valuesd& values,
                        int degree) {
  polynomial::monomial_basis<Dim> basis(degree);

  Eigen::MatrixXd at = basis.evaluate(points, grad_points);

  common::valuesd coeffs = (at * at.transpose()).inverse() * at * values;

  return values - at.transpose() * coeffs;
}

template <int Dim>
common::valuesd detrend(const geometry::pointsNd<Dim>& points, const common::valuesd& values,
                        int degree) {
  return detrend(points, geometry::pointsNd<Dim>(0, Dim), values, degree);
}

}  // namespace polatory::kriging
