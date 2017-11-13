// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include <polatory/polynomial/basis_base.hpp>
#include <polatory/polynomial/monomial_basis.hpp>

namespace polatory {
namespace polynomial {

template <class Floating = double>
class orthonormal_basis : public basis_base {
  using Vector3F = Eigen::Matrix<Floating, 3, 1>;
  using VectorXF = Eigen::Matrix<Floating, Eigen::Dynamic, 1>;
  using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;

  monomial_basis<Floating> mono_basis;

  MatrixXF c_hat;

public:
  template <class Container>
  orthonormal_basis(int dimension, int degree, const Container& points)
    : basis_base(dimension, degree)
    , mono_basis(dimension, degree) {
    auto pt = mono_basis.evaluate_points(points);
    auto u_hat = MatrixXF(pt.rows(), pt.cols());

    auto size = basis_size();

    c_hat = MatrixXF::Identity(size, size);

    VectorXF u = pt.row(0);
    auto u_norm = u.norm();
    c_hat(0, 0) /= u_norm;
    u_hat.row(0) = u / u_norm;

    for (size_t i = 1; i < size; i++) {
      u = pt.row(i);
      for (size_t j = 0; j < i; j++) {
        for (size_t k = j; k < i; k++) {
          c_hat(i, j) -= u_hat.row(k).dot(pt.row(i)) * c_hat(k, j);
        }
        u += c_hat(i, j) * pt.row(j);
      }
      u_norm = u.norm();
      c_hat.row(i) /= u_norm;
      u_hat.row(i) = u / u_norm;
    }
  }

  template <class Container>
  MatrixXF evaluate_points(const Container& points) const {
    auto pt = mono_basis.evaluate_points(points);

    return c_hat * pt;
  }
};

} // namespace polynomial
} // namespace polatory
