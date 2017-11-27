// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>
#include <Eigen/LU>

#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/polynomial/monomial_basis.hpp>

namespace polatory {
namespace polynomial {

template <class Floating = double>
class lagrange_basis : public basis_base {
  using Vector3F = Eigen::Matrix<Floating, 3, 1>;
  using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;

public:
  lagrange_basis(int dimension, int degree, const geometry::points3d& points)
    : basis_base(dimension, degree)
    , mono_basis_(dimension, degree) {
    assert(points.rows() == basis_size());

    auto pt = mono_basis_.evaluate_points(points);

    coeffs_ = pt.transpose().fullPivLu().inverse();
  }

  MatrixXF evaluate_points(const geometry::points3d& points) const {
    auto pt = mono_basis_.evaluate_points(points);

    return coeffs_.transpose() * pt;
  }

private:
  const monomial_basis<Floating> mono_basis_;

  MatrixXF coeffs_;
};

} // namespace polynomial
} // namespace polatory
