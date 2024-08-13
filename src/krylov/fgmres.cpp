#include <polatory/krylov/fgmres.hpp>

namespace polatory::krylov {

Fgmres::Fgmres(const LinearOperator& op, const VecX& rhs, Index max_iter)
    : Gmres(op, rhs, max_iter) {}

VecX Fgmres::solution_vector() const {
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

  VecX x = x0_;
  for (Index i = 0; i < iter_; i++) {
    x += y(i) * zs_.at(i);
  }

  return x;
}

void Fgmres::add_preconditioned_krylov_basis(const VecX& z) { zs_.push_back(z); }

}  // namespace polatory::krylov
