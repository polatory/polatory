// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace polynomial {

class lagrange_basis : public polynomial_basis_base {
public:
  lagrange_basis(int dimension, int degree, const geometry::points3d& points)
    : polynomial_basis_base(dimension, degree)
    , mono_basis_(dimension, degree) {
    POLATORY_ASSERT(static_cast<index_t>(points.rows()) == basis_size());

    auto pt = mono_basis_.evaluate_points(points);

    auto lu_of_p = pt.transpose().fullPivLu();
    if (!lu_of_p.isInvertible()) {
      throw std::domain_error("The set of points is not unisolvent.");
    }

    coeffs_ = lu_of_p.inverse();
  }

  Eigen::MatrixXd evaluate_points(const geometry::points3d& points) const {
    auto pt = mono_basis_.evaluate_points(points);

    return coeffs_.transpose() * pt;
  }

private:
  const monomial_basis mono_basis_;

  Eigen::MatrixXd coeffs_;
};

}  // namespace polynomial
}  // namespace polatory
