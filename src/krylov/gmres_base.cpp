// Copyright (c) 2016, GSI and The Polatory Authors.

#include "krylov/gmres_base.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace polatory {
namespace krylov {

bool gmres_base::print_progress = true;

double gmres_base::absolute_residual() const
{
   return std::abs(g_(iter_));
}

bool gmres_base::converged() const
{
   return converged_;
}

int gmres_base::iteration_count() const
{
   return iter_;
}

int gmres_base::max_iterations() const
{
   return max_iter_;
}

double gmres_base::relative_residual() const
{
   return std::abs(g_(iter_)) / rhs_norm_;
}

void gmres_base::set_left_preconditioner(const linear_operator &left_preconditioner)
{
   assert(left_preconditioner.size() == m_);

   this->left_pc_ = &left_preconditioner;
}

void gmres_base::set_initial_solution(const Eigen::VectorXd& x0)
{
   assert(x0.size() == m_);

   this->x0_ = x0;
}

void gmres_base::set_right_preconditioner(const linear_operator &right_preconditioner)
{
   assert(right_preconditioner.size() == m_);

   this->right_pc_ = &right_preconditioner;
}

void gmres_base::setup()
{
   c_ = Eigen::VectorXd::Zero(max_iter_);
   s_ = Eigen::VectorXd::Zero(max_iter_);

   g_ = Eigen::VectorXd::Zero(max_iter_ + 1);

   Eigen::VectorXd r0;
   if (x0_.isZero()) {
      r0 = left_preconditioned(rhs_);
   } else {
      r0 = left_preconditioned(rhs_ - op_(x0_));
   }
   g_(0) = r0.norm();
   vs_.push_back(r0 / g_(0));

   r_ = Eigen::MatrixXd::Zero(max_iter_ + 1, max_iter_);
}

Eigen::VectorXd gmres_base::solution_vector() const
{
   // r is an upper triangular matrix.
   // Perform backward substitution to solve r y == g for y.
   Eigen::VectorXd y = Eigen::VectorXd::Zero(iter_);
   for (int j = iter_ - 1; j >= 0; j--) {
      y(j) = g_(j);
      for (int i = j + 1; i <= iter_ - 1; i++) {
         y(j) -= r_(j, i) * y(i);
      }
      y(j) /= r_(j, j);
   }

   Eigen::VectorXd x = Eigen::VectorXd::Zero(m_);
   for (int i = 0; i < iter_; i++) {
      x += y(i) * vs_[i];
   }
   x = right_preconditioned(x);
   x += x0_;

   return x;
}

void gmres_base::solve(double tolerance)
{
   for (; iter_ < max_iter_;) {
      if (print_progress)
         std::cout << iter_ << ": \t" << relative_residual() << std::endl;
      iterate_process();
      if (relative_residual() < tolerance) {
         converged_ = true;
         break;
      }
   }
   if (print_progress)
      std::cout << iter_ << ": \t" << relative_residual() << std::endl;
}

gmres_base::gmres_base(const linear_operator &op, const Eigen::VectorXd &rhs, int max_iter)
   : op_(op)
   , m_(rhs.size())
   , max_iter_(max_iter)
   , x0_(Eigen::VectorXd::Zero(m_))
   , left_pc_(nullptr)
   , right_pc_(nullptr)
   , iter_(0)
   , rhs_(rhs)
   , rhs_norm_(rhs.norm())
   , converged_(false)
{
}

Eigen::VectorXd gmres_base::left_preconditioned(const Eigen::VectorXd x) const
{
   return left_pc_ != nullptr
      ? (*left_pc_)(x)
      : x;
}

Eigen::VectorXd gmres_base::right_preconditioned(const Eigen::VectorXd x) const
{
   return right_pc_ != nullptr
      ? (*right_pc_)(x)
      : x;
}

} // namespace krylov
} // namespace polatory
