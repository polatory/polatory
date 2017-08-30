// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

#include "monomial_basis.hpp"
#include "basis_base.hpp"

namespace polatory {
namespace polynomial {

template<typename Floating = double>
class lagrange_basis : public basis_base {
   using Vector3F = Eigen::Matrix<Floating, 3, 1>;
   using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;

   monomial_basis<Floating> mono_basis;

   MatrixXF coeffs;

public:
   template<typename Container>
   lagrange_basis(int degree, const Container& points)
      : basis_base(degree)
      , mono_basis(degree)
   {
      auto pt = mono_basis.evaluate_points(points);

      auto dim = dimension();
      MatrixXF rhs = MatrixXF::Identity(dim, dim);

      coeffs = pt.transpose().fullPivLu().solve(rhs);
   }

   template<typename Container>
   MatrixXF evaluate_points(const Container& points) const
   {
      auto pt = mono_basis.evaluate_points(points);

      return coeffs.transpose() * pt;
   }
};

} // namespace polynomial
} // namespace polatory
