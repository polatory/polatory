// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "gmres_base.hpp"

namespace polatory {
namespace krylov {

class minres : public gmres_base {
   double beta;
   
public:
   minres(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter)
      : gmres_base(op, rhs, max_iter)
      , beta(0.0)
   {
   }

   void iterate_process() override
   {
      if (iter == max_iter) return;

      int j = iter;

      // Lanczos process
      vs.push_back(left_preconditioned(op(right_preconditioned(vs[j]))));
      r(j, j) = vs[j].dot(vs[j + 1]);
      if (j == 0) {
         vs[j + 1] -= r(j, j) * vs[j];
      } else {
         r(j - 1, j) = beta;  // beta_{j - 1}
         vs[j + 1] -= r(j - 1, j) * vs[j - 1] + r(j, j) * vs[j];
      }
      r(j + 1, j) = vs[j + 1].norm();
      beta = r(j + 1, j);     // beta_j
      vs[j + 1] /= r(j + 1, j);

      // Update matrix R by Givens rotation
      for (int i = (std::max)(0, j - 2); i < j; i++) {
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
