#include <cmath>
#include <polatory/krylov/minres.hpp>

namespace polatory {
namespace krylov {

minres::minres(const linear_operator& op, const common::valuesd& rhs, index_t max_iter)
    : gmres_base(op, rhs, max_iter), beta_(0.0) {}

void minres::iterate_process() {
  if (iter_ == max_iter_) return;

  auto j = iter_;

  // Lanczos process
  vs_.push_back(left_preconditioned(op_(right_preconditioned(vs_[j]))));
  r_(j, j) = vs_[j].dot(vs_[j + 1]);
  if (j == 0) {
    vs_[j + 1] -= r_(j, j) * vs_[j];
  } else {
    r_(j - 1, j) = beta_;  // beta_{j - 1}
    vs_[j + 1] -= r_(j - 1, j) * vs_[j - 1] + r_(j, j) * vs_[j];
  }
  r_(j + 1, j) = vs_[j + 1].norm();
  beta_ = r_(j + 1, j);  // beta_j
  vs_[j + 1] /= r_(j + 1, j);

  // Update matrix R by Givens rotation
  for (index_t i = std::max(index_t{0}, j - 2); i < j; i++) {
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

}  // namespace krylov
}  // namespace polatory
