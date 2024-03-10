#pragma once

#include <Eigen/Core>

namespace polatory::numeric {

template <int P = 2, class DerivedApprox, class DerivedExact>
double absolute_error(const Eigen::MatrixBase<DerivedApprox>& approx,
                      const Eigen::MatrixBase<DerivedExact>& exact) {
  return (approx - exact).template lpNorm<P>();
}

template <int P = 2, class DerivedApprox, class DerivedExact>
double relative_error(const Eigen::MatrixBase<DerivedApprox>& approx,
                      const Eigen::MatrixBase<DerivedExact>& exact) {
  return (approx - exact).template lpNorm<P>() / exact.template lpNorm<P>();
}

}  // namespace polatory::numeric
