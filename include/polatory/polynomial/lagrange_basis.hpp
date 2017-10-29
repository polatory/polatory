// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>
#include <Eigen/LU>

#include "monomial_basis.hpp"
#include "basis_base.hpp"

namespace polatory {
namespace polynomial {

template <class Floating = double>
class lagrange_basis : public basis_base {
  using Vector3F = Eigen::Matrix<Floating, 3, 1>;
  using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;

  monomial_basis <Floating> mono_basis;

  MatrixXF coeffs;

public:
  template <class Container>
  lagrange_basis(int dimension, int degree, const Container& points)
    : basis_base(dimension, degree)
    , mono_basis(dimension, degree) {
    auto size = basis_size();
    assert(points.size() == size);

    auto pt = mono_basis.evaluate_points(points);

    coeffs = pt.transpose().fullPivLu().inverse();
  }

  template <class Container>
  MatrixXF evaluate_points(const Container& points) const {
    auto pt = mono_basis.evaluate_points(points);

    return coeffs.transpose() * pt;
  }
};

} // namespace polynomial
} // namespace polatory
