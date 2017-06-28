// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "gmres_base.hpp"

namespace polatory {
namespace krylov {

class gmres : public gmres_base {
public:
   gmres(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter)
      : gmres_base(op, rhs, max_iter)
   {
   }

   void iterate_process() override
   {
      if (iter == max_iter) return;

      int j = iter;

      // Arnoldi process
      auto z = right_preconditioned(vs[j]);
      add_preconditioned_krylov_basis(z);
      vs.push_back(left_preconditioned(op(z)));
#pragma omp parallel for
      for (int i = 0; i <= j; i++) {
         r(i, j) = vs[i].dot(vs[j + 1]);
      }
      for (int i = 0; i <= j; i++) {
         vs[j + 1] -= r(i, j) * vs[i];
      }
      r(j + 1, j) = vs[j + 1].norm();
      vs[j + 1] /= r(j + 1, j);

      // Update matrix R by Givens rotation
      for (int i = 0; i < j; i++) {
         double x = r(i, j);
         double y = r(i + 1, j);
         double tmp1 = c(i) * x + s(i) * y;
         double tmp2 = -s(i) * x + c(i) * y;
         r(i, j) = tmp1;
         r(i + 1, j) = tmp2;
      }
      double x = r(j, j);
      double y = r(j + 1, j);
      double den = std::hypot(x, y);
      c(j) = x / den;
      s(j) = y / den;

      r(j, j) = c(j) * x + s(j) * y;
      g(j + 1) = -s(j) * g(j);
      g(j) = c(j) * g(j);

      iter++;
   }
};

} // namespace krylov
} // namespace polatory
