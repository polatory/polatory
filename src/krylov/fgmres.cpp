#include <polatory/krylov/fgmres.hpp>

namespace polatory::krylov {

fgmres::fgmres(const linear_operator& op, const vectord& rhs, index_t max_iter)
    : gmres(op, rhs, max_iter) {}

vectord fgmres::solution_vector() const {
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

  vectord x = x0_;
  for (index_t i = 0; i < iter_; i++) {
    x += y(i) * zs_.at(i);
  }

  return x;
}

void fgmres::add_preconditioned_krylov_basis(const vectord& z) { zs_.push_back(z); }

}  // namespace polatory::krylov
