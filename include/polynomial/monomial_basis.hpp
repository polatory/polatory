// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

#include <Eigen/Core>

#include "basis_base.hpp"

namespace polatory {
namespace polynomial {

template<typename Floating = double>
class monomial_basis : public basis_base {
   using Vector3F = Eigen::Matrix<Floating, 3, 1>;
   using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;

public:
   explicit monomial_basis(int degree)
      : basis_base(degree)
   {
      assert(degree >= 0 && degree <= 2);
   }

   template<typename Container>
   MatrixXF evaluate_points(const Container& points) const
   {
      size_t n_points = points.size();

      MatrixXF result = MatrixXF(dimension(), n_points);

      switch (degree()) {
      case 0:
         // 1
         for (size_t i = 0; i < n_points; i++) {
            result(0, i) = 1.0;
         }
         break;
      case 1:
         // 1, x, y, z
         for (size_t i = 0; i < n_points; i++) {
            const auto& p = points[i];

            result(0, i) = 1.0;
            result(1, i) = p[0];
            result(2, i) = p[1];
            result(3, i) = p[2];
         }
         break;
      case 2:
         // 1, x, y, z, x^2, xy, xz, y^2, yz, z^2
         for (size_t i = 0; i < n_points; i++) {
            const auto& p = points[i];

            result(0, i) = 1.0;
            result(1, i) = p[0];
            result(2, i) = p[1];
            result(3, i) = p[2];
            result(4, i) = p[0] * p[0];
            result(5, i) = p[0] * p[1];
            result(6, i) = p[0] * p[2];
            result(7, i) = p[1] * p[1];
            result(8, i) = p[1] * p[2];
            result(9, i) = p[2] * p[2];
         }
         break;
      default:
         assert(false);
         break;
      }

      return result;
   }
};

} // namespace polynomial
} // namespace polatory
