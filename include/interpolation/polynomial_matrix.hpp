// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>

namespace polatory {
namespace interpolation {

// Represents and operates the polynomial part of a RBF operator:
//     -     -
//    |  O  P |
//    |       |
//    | P^T O |
//     -     -
// where P_ij = p_j(x_i) and {p_j} are the monomial basis:
//  degree                    basis
// -------------------------------------------------
//     0      1
//     1      1, x, y, z
//     2      1, x, y, z, x^2, xy, xz, y^2, yz, z^2
template<class Basis>
class polynomial_matrix {
   Basis basis;

   // Transposed polynomial matrix P^T.
   Eigen::MatrixXd pt;

public:
   polynomial_matrix(int degree)
      : basis(degree)
   {
   }

   // Returns a vector consists of two parts:
   //   0...m-1   : P c
   //   m...m+l-1 : P^T lambda
   // where m is the number of points and l is the size of the basis.
   template<typename Derived>
   Eigen::VectorXd evaluate(const Eigen::MatrixBase<Derived>& lambda_c) const
   {
      auto l = pt.rows();
      auto m = pt.cols();

      assert(lambda_c.size() == m + l);

      Eigen::VectorXd output(m + l);

      auto lambda = lambda_c.head(m);
      auto c = lambda_c.tail(l);

      output.head(m) = pt.transpose() * c;
      output.tail(l) = pt * lambda;

      return output;
   }

   template<typename Container>
   void set_points(const Container& points)
   {
      pt = basis.evaluate_points(points);
   }
};

} // namespace interpolation
} // namespace polatory
