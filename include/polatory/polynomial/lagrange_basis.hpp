#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::polynomial {

template <int Dim>
class LagrangeBasis : public PolynomialBasisBase<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = PolynomialBasisBase<kDim>;
  using MonomialBasis = MonomialBasis<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  using Base::basis_size;
  using Base::degree;

  template <class Derived>
  LagrangeBasis(int degree, const Eigen::MatrixBase<Derived>& points)
      : LagrangeBasis(degree, points, Points(0, kDim)) {}

  template <class DerivedPoints, class DerivedGradPoints>
  LagrangeBasis(int degree, const Eigen::MatrixBase<DerivedPoints>& points,
                const Eigen::MatrixBase<DerivedGradPoints>& grad_points)
      : Base(degree), mono_basis_(degree) {
    POLATORY_ASSERT(points.rows() == basis_size() ||
                    degree == 1 && points.rows() == 1 && grad_points.rows() == 1);

    auto p = mono_basis_.evaluate(points, grad_points);

    Eigen::FullPivLU<MatX> lu(p);

    if (!lu.isInvertible()) {
      throw std::invalid_argument("the set of points is not unisolvent");
    }

    coeffs_ = lu.inverse();
    rcond_ = lu.rcond();
  }

  template <class Derived>
  MatX evaluate(const Eigen::MatrixBase<Derived>& points) const {
    return evaluate(points, Points(0, kDim));
  }

  template <class DerivedPoints, class DerivedGradPoints>
  MatX evaluate(const Eigen::MatrixBase<DerivedPoints>& points,
                const Eigen::MatrixBase<DerivedGradPoints>& grad_points) const {
    auto p = mono_basis_.evaluate(points, grad_points);

    return p * coeffs_;
  }

  double rcond() const { return rcond_; }

 private:
  const MonomialBasis mono_basis_;

  MatX coeffs_;
  double rcond_{};
};

}  // namespace polatory::polynomial
