#include <cmath>
#include <polatory/common/macros.hpp>
#include <polatory/krylov/gmres_base.hpp>

namespace polatory::krylov {

double gmres_base::absolute_residual() const { return std::abs(g_(iter_)); }

bool gmres_base::converged() const { return converged_; }

index_t gmres_base::iteration_count() const { return iter_; }

index_t gmres_base::max_iterations() const { return max_iter_; }

double gmres_base::relative_residual() const { return std::abs(g_(iter_)) / rhs_norm_; }

void gmres_base::set_left_preconditioner(const linear_operator& left_preconditioner) {
  POLATORY_ASSERT(left_preconditioner.size() == m_);

  left_pc_ = &left_preconditioner;
}

void gmres_base::set_initial_solution(const vectord& x0) {
  POLATORY_ASSERT(x0.rows() == m_);

  x0_ = x0;
}

void gmres_base::set_right_preconditioner(const linear_operator& right_preconditioner) {
  POLATORY_ASSERT(right_preconditioner.size() == m_);

  right_pc_ = &right_preconditioner;
}

void gmres_base::setup() {
  c_ = vectord::Zero(max_iter_);
  s_ = vectord::Zero(max_iter_);

  g_ = vectord::Zero(max_iter_ + 1);

  vectord r0 = x0_.isZero() ? rhs_ : rhs_ - op_(x0_);
  r0 = left_preconditioned(r0);
  g_(0) = r0.norm();
  vs_.emplace_back(r0 / g_(0));

  r_ = matrixd::Zero(max_iter_ + 1, max_iter_);
}

vectord gmres_base::solution_vector() const {
  // r is an upper triangular matrix.
  // Perform backward substitution to solve r y == g for y.
  vectord y = vectord::Zero(iter_);
  for (index_t j = iter_ - 1; j >= 0; j--) {
    y(j) = g_(j);
    for (index_t i = j + 1; i <= iter_ - 1; i++) {
      y(j) -= r_(j, i) * y(i);
    }
    y(j) /= r_(j, j);
  }

  vectord x = vectord::Zero(m_);
  for (index_t i = 0; i < iter_; i++) {
    x += y(i) * vs_.at(i);
  }
  x = right_preconditioned(x);
  x += x0_;

  return x;
}

gmres_base::gmres_base(const linear_operator& op, const vectord& rhs, index_t max_iter)
    : op_(op),
      m_(rhs.size()),
      max_iter_(max_iter),
      x0_(vectord::Zero(m_)),
      rhs_(rhs),
      rhs_norm_(rhs.norm()) {}

vectord gmres_base::left_preconditioned(const vectord& x) const {
  return left_pc_ != nullptr ? (*left_pc_)(x) : x;
}

vectord gmres_base::right_preconditioned(const vectord& x) const {
  return right_pc_ != nullptr ? (*right_pc_)(x) : x;
}

}  // namespace polatory::krylov
