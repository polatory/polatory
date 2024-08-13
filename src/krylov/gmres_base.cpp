#include <cmath>
#include <polatory/common/macros.hpp>
#include <polatory/krylov/gmres_base.hpp>

namespace polatory::krylov {

double GmresBase::absolute_residual() const { return std::abs(g_(iter_)); }

bool GmresBase::converged() const { return converged_; }

Index GmresBase::iteration_count() const { return iter_; }

Index GmresBase::max_iterations() const { return max_iter_; }

double GmresBase::relative_residual() const { return std::abs(g_(iter_)) / rhs_norm_; }

void GmresBase::set_left_preconditioner(const LinearOperator& left_preconditioner) {
  POLATORY_ASSERT(left_preconditioner.size() == m_);

  left_pc_ = &left_preconditioner;
}

void GmresBase::set_initial_solution(const VecX& x0) {
  POLATORY_ASSERT(x0.rows() == m_);

  x0_ = x0;
}

void GmresBase::set_right_preconditioner(const LinearOperator& right_preconditioner) {
  POLATORY_ASSERT(right_preconditioner.size() == m_);

  right_pc_ = &right_preconditioner;
}

void GmresBase::setup() {
  c_ = VecX::Zero(max_iter_);
  s_ = VecX::Zero(max_iter_);

  g_ = VecX::Zero(max_iter_ + 1);

  VecX r0 = x0_.isZero() ? rhs_ : rhs_ - op_(x0_);
  r0 = left_preconditioned(r0);
  g_(0) = r0.norm();
  vs_.emplace_back(r0 / g_(0));

  r_ = MatX::Zero(max_iter_ + 1, max_iter_);
}

VecX GmresBase::solution_vector() const {
  // r is an upper triangular matrix.
  // Perform backward substitution to solve r y == g for y.
  VecX y = VecX::Zero(iter_);
  for (Index j = iter_ - 1; j >= 0; j--) {
    y(j) = g_(j);
    for (Index i = j + 1; i <= iter_ - 1; i++) {
      y(j) -= r_(j, i) * y(i);
    }
    y(j) /= r_(j, j);
  }

  VecX x = VecX::Zero(m_);
  for (Index i = 0; i < iter_; i++) {
    x += y(i) * vs_.at(i);
  }
  x = right_preconditioned(x);
  x += x0_;

  return x;
}

GmresBase::GmresBase(const LinearOperator& op, const VecX& rhs, Index max_iter)
    : op_(op),
      m_(rhs.size()),
      max_iter_(max_iter),
      x0_(VecX::Zero(m_)),
      rhs_(rhs),
      rhs_norm_(rhs.norm()) {}

VecX GmresBase::left_preconditioned(const VecX& x) const {
  return left_pc_ != nullptr ? (*left_pc_)(x) : x;
}

VecX GmresBase::right_preconditioned(const VecX& x) const {
  return right_pc_ != nullptr ? (*right_pc_)(x) : x;
}

}  // namespace polatory::krylov
