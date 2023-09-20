#pragma once

#include <Eigen/Core>
#include <polatory/types.hpp>

namespace polatory::common {

template <class Derived>
void orthonormalize_cols(Eigen::MatrixBase<Derived>& m) {
  // The (modified) Gram-Schmidt orthonormalization.
  auto n = m.cols();
  for (index_t i = 0; i < n; i++) {
    m.col(i) /= m.col(i).norm();
    for (index_t j = i + 1; j < n; j++) {
      m.col(j) -= m.col(i).dot(m.col(j)) * m.col(i);
    }
  }
}

}  // namespace polatory::common
