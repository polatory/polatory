// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/krylov/fgmres.hpp>

namespace polatory {
namespace krylov {

fgmres::fgmres(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter)
  : gmres(op, rhs, max_iter) {
}

Eigen::VectorXd fgmres::solution_vector() const {
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

  Eigen::VectorXd x = x0_;
  for (int i = 0; i < iter_; i++) {
    x += y(i) * zs_[i];
  }

  return x;
}

void fgmres::add_preconditioned_krylov_basis(const Eigen::VectorXd& z) {
  zs_.push_back(z);
}

} // namespace krylov
} // namespace polatory
