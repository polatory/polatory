// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Core>

#include "linear_operator.hpp"

namespace polatory {
namespace krylov {

class gmres_base {
protected:
   const linear_operator& op;

   // Dimension.
   const size_t m;

   // Maximum # of iteration.
   const int max_iter;

   // Initial solution.
   Eigen::VectorXd x0;

   // Left preconditioner.
   const linear_operator *left_pc;

   // Right preconditioner.
   const linear_operator *right_pc;

   // Current # of iteration.
   int iter;

   // Constant (right-hand side) vector.
   const Eigen::VectorXd rhs;

   // L2 norm of rhs.
   double rhs_norm;

   // Orthonormal basis vectors for the Krylov subspace.
   std::vector<Eigen::VectorXd> vs;

   // Upper triangular matrix of QR decomposition.
   Eigen::MatrixXd r;

   // Cosines for the Givens rotations.
   Eigen::VectorXd c;

   // Sines for the Givens rotations.
   Eigen::VectorXd s;

   // Sequence of residuals.
   Eigen::VectorXd g;

   bool m_converged;

   gmres_base(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter)
      : op(op)
      , m(rhs.size())
      , max_iter(max_iter)
      , x0(Eigen::VectorXd::Zero(m))
      , left_pc(nullptr)
      , right_pc(nullptr)
      , iter(0)
      , rhs(rhs)
      , rhs_norm(rhs.norm())
      , m_converged(false)
   {
   }

   virtual ~gmres_base() {}

   virtual void add_preconditioned_krylov_basis(const Eigen::VectorXd& z)
   {
   }

   Eigen::VectorXd left_preconditioned(const Eigen::VectorXd x) const
   {
      return left_pc != nullptr
         ? (*left_pc)(x)
         : x;
   }

   Eigen::VectorXd right_preconditioned(const Eigen::VectorXd x) const
   {
      return right_pc != nullptr
         ? (*right_pc)(x)
         : x;
   }

public:
   static bool print_progress;

   double absolute_residual() const
   {
      return std::abs(g(iter));
   }

   bool converged() const
   {
      return m_converged;
   }

   virtual void iterate_process() = 0;

   int iteration_count() const
   {
      return iter;
   }

   int max_iterations() const
   {
      return max_iter;
   }

   virtual void set_left_preconditioner(const linear_operator& left_preconditioner)
   {
      assert(left_preconditioner.size() == m);

      this->left_pc = &left_preconditioner;
   }

   template<typename Derived>
   void set_initial_solution(const Eigen::MatrixBase<Derived>& x0)
   {
      assert(x0.size() == m);

      this->x0 = x0;
   }

   virtual void set_right_preconditioner(const linear_operator& right_preconditioner)
   {
      assert(right_preconditioner.size() == m);

      this->right_pc = &right_preconditioner;
   }

   virtual void setup()
   {
      c = Eigen::VectorXd::Zero(max_iter);
      s = Eigen::VectorXd::Zero(max_iter);

      g = Eigen::VectorXd::Zero(max_iter + 1);

      Eigen::VectorXd r0;
      if (x0.isZero()) {
         r0 = left_preconditioned(rhs);
      } else {
         r0 = left_preconditioned(rhs - op(x0));
      }
      g(0) = r0.norm();
      vs.push_back(r0 / g(0));

      r = Eigen::MatrixXd::Zero(max_iter + 1, max_iter);
   }

   double relative_residual() const
   {
      return std::abs(g(iter)) / rhs_norm;
   }

   virtual Eigen::VectorXd solution_vector() const
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

      Eigen::VectorXd x = Eigen::VectorXd::Zero(m);
      for (int i = 0; i < iter; i++) {
         x += y(i) * vs[i];
      }
      x = right_preconditioned(x);
      x += x0;

      return x;
   }

   // tolerance: Tolerance of the relative residual (stopping criterion).
   void solve(double tolerance)
   {
      for (; iter < max_iter;) {
         if (print_progress)
            std::cout << iter << ": \t" << relative_residual() << std::endl;
         iterate_process();
         if (relative_residual() < tolerance) {
            m_converged = true;
            break;
         }
      }
      if (print_progress)
         std::cout << iter << ": \t" << relative_residual() << std::endl;
   }
};

} // namespace krylov
} // namespace polatory
