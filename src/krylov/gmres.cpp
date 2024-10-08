#include <cmath>
#include <polatory/krylov/gmres.hpp>

namespace polatory::krylov {

Gmres::Gmres(const LinearOperator& op, const VecX& rhs, Index max_iter)
    : GmresBase(op, rhs, max_iter) {}

void Gmres::iterate_process() {
  if (iter_ == max_iter_) {
    return;
  }

  auto j = iter_;

  // Arnoldi process
  auto z = right_preconditioned(vs_.at(j));
  add_preconditioned_krylov_basis(z);
  vs_.push_back(left_preconditioned(op_(z)));
#pragma omp parallel for schedule(static)
  for (Index i = 0; i <= j; i++) {
    r_(i, j) = vs_.at(i).dot(vs_.at(j + 1));
  }
  for (Index i = 0; i <= j; i++) {
    vs_.at(j + 1) -= r_(i, j) * vs_.at(i);
  }
  r_(j + 1, j) = vs_.at(j + 1).norm();
  vs_.at(j + 1) /= r_(j + 1, j);

  // Update matrix R by Givens rotation
  for (Index i = 0; i < j; i++) {
    auto x = r_(i, j);
    auto y = r_(i + 1, j);
    auto tmp1 = c_(i) * x + s_(i) * y;
    auto tmp2 = -s_(i) * x + c_(i) * y;
    r_(i, j) = tmp1;
    r_(i + 1, j) = tmp2;
  }
  auto x = r_(j, j);
  auto y = r_(j + 1, j);
  auto den = std::hypot(x, y);
  c_(j) = x / den;
  s_(j) = y / den;

  r_(j, j) = c_(j) * x + s_(j) * y;
  g_(j + 1) = -s_(j) * g_(j);
  g_(j) = c_(j) * g_(j);

  iter_++;
}

}  // namespace polatory::krylov
