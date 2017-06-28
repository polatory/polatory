// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "common/exception.hpp"
#include "gmres.hpp"

namespace polatory {
namespace krylov {

class fgmres : public gmres {
   // zs[i] := right_preconditioned(vs[i - 1]).
   std::vector<Eigen::VectorXd> zs;

   void add_preconditioned_krylov_basis(const Eigen::VectorXd& z) override
   {
      zs.push_back(z);
   }

public:
   fgmres(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter)
      : gmres(op, rhs, max_iter)
   {
   }

   void set_left_preconditioner(const linear_operator& left_preconditioner) override
   {
      throw common::unsupported_method("set_left_preconditioner");
   }

   Eigen::VectorXd solution_vector() const override
   {
      // r is an upper triangular matrix.
      // Perform backward substitution to solve r y == g for y.
      Eigen::VectorXd y = Eigen::VectorXd::Zero(iter);
      for (int j = iter - 1; j >= 0; j--) {
         y(j) = g(j);
         for (int i = j + 1; i <= iter - 1; i++) {
            y(j) -= r(j, i) * y(i);
         }
         y(j) /= r(j, j);
      }

      Eigen::VectorXd x = x0;
      for (int i = 0; i < iter; i++) {
         x += y(i) * zs[i];
      }

      return x;
   }
};

} // namespace krylov
} // namespace polatory
