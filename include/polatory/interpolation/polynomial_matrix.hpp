// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace interpolation {

// Represents and operates the polynomial part of a RBF operator:
//     -     -
//    |  O  P |
//    |       |
//    | P^T O |
//     -     -
// where P_ij = p_j(x_i) and {p_j} are the monomial basis.
template <class Basis>
class polynomial_matrix {
public:
  polynomial_matrix(int dimension, int degree)
    : basis_(dimension, degree) {
  }

  // Returns a vector consists of two parts:
  //   0...m-1   : P c
  //   m...m+l-1 : P^T lambda
  // where m is the number of points and l is the size of the basis.
  template <class Derived>
  common::valuesd evaluate(const Eigen::MatrixBase<Derived>& lambda_c) const {
    auto l = pt_.rows();
    auto m = pt_.cols();

    assert(lambda_c.rows() == m + l);

    common::valuesd output(m + l);

    auto lambda = lambda_c.head(m);
    auto c = lambda_c.tail(l);

    output.head(m) = pt_.transpose() * c;
    output.tail(l) = pt_ * lambda;

    return output;
  }

  void set_points(const geometry::points3d& points) {
    pt_ = basis_.evaluate_points(points);
  }

private:
  Basis basis_;

  // Transposed polynomial matrix P^T.
  Eigen::MatrixXd pt_;
};

}  // namespace interpolation
}  // namespace polatory
