#include <cmath>
#include <iostream>
#include <polatory/common/macros.hpp>
#include <polatory/krylov/gmres_base.hpp>

namespace polatory::krylov {

bool gmres_base::print_progress = true;

double gmres_base::absolute_residual() const { return std::abs(g_(iter_)); }

bool gmres_base::converged() const { return converged_; }

index_t gmres_base::iteration_count() const { return iter_; }

index_t gmres_base::max_iterations() const { return max_iter_; }

double gmres_base::relative_residual() const { return std::abs(g_(iter_)) / rhs_norm_; }

void gmres_base::set_left_preconditioner(const linear_operator& left_preconditioner) {
  POLATORY_ASSERT(left_preconditioner.size() == m_);

  left_pc_ = &left_preconditioner;
}

void gmres_base::set_initial_solution(const common::valuesd& x0) {
  POLATORY_ASSERT(static_cast<index_t>(x0.rows()) == m_);

  x0_ = x0;
}

void gmres_base::set_right_preconditioner(const linear_operator& right_preconditioner) {
  POLATORY_ASSERT(right_preconditioner.size() == m_);

  right_pc_ = &right_preconditioner;
}

void gmres_base::setup() {
  c_ = common::valuesd::Zero(max_iter_);
  s_ = common::valuesd::Zero(max_iter_);

  g_ = common::valuesd::Zero(max_iter_ + 1);

  common::valuesd r0;
  if (x0_.isZero()) {
    r0 = left_preconditioned(rhs_);
  } else {
    r0 = left_preconditioned(rhs_ - op_(x0_));
  }
  g_(0) = r0.norm();
  vs_.emplace_back(r0 / g_(0));

  r_ = Eigen::MatrixXd::Zero(max_iter_ + 1, max_iter_);
}

common::valuesd gmres_base::solution_vector() const {
  // r is an upper triangular matrix.
  // Perform backward substitution to solve r y == g for y.
  common::valuesd y = common::valuesd::Zero(iter_);
  for (index_t j = iter_ - 1; j >= 0; j--) {
    y(j) = g_(j);
    for (index_t i = j + 1; i <= iter_ - 1; i++) {
      y(j) -= r_(j, i) * y(i);
    }
    y(j) /= r_(j, j);
  }

  common::valuesd x = common::valuesd::Zero(m_);
  for (index_t i = 0; i < iter_; i++) {
    x += y(i) * vs_[i];
  }
  x = right_preconditioned(x);
  x += x0_;

  return x;
}

void gmres_base::solve(double tolerance) {
  for (; iter_ < max_iter_;) {
    if (print_progress) {
      std::cout << iter_ << ": \t" << relative_residual() << std::endl;
    }
    iterate_process();
    if (relative_residual() < tolerance) {
      converged_ = true;
      break;
    }
  }
  if (print_progress) {
    std::cout << iter_ << ": \t" << relative_residual() << std::endl;
  }
}

gmres_base::gmres_base(const linear_operator& op, const common::valuesd& rhs, index_t max_iter)
    : op_(op),
      m_(static_cast<index_t>(rhs.rows())),
      max_iter_(max_iter),
      x0_(common::valuesd::Zero(m_)),
      left_pc_(nullptr),
      right_pc_(nullptr),
      iter_(0),
      rhs_(rhs),
      rhs_norm_(rhs.norm()),
      converged_(false) {}

common::valuesd gmres_base::left_preconditioned(const common::valuesd& x) const {
  return left_pc_ != nullptr ? (*left_pc_)(x) : x;
}

common::valuesd gmres_base::right_preconditioned(const common::valuesd& x) const {
  return right_pc_ != nullptr ? (*right_pc_)(x) : x;
}

}  // namespace polatory::krylov
