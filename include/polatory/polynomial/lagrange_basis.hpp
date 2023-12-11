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
class lagrange_basis : public polynomial_basis_base<Dim> {
  using Base = polynomial_basis_base<Dim>;
  using Points = geometry::pointsNd<Dim>;
  using MonomialBasis = monomial_basis<Dim>;

 public:
  using Base::basis_size;
  using Base::degree;

  template <class Derived>
  lagrange_basis(int degree, const Eigen::MatrixBase<Derived>& points)
      : lagrange_basis(degree, points, Points(0, Dim)) {}

  template <class DerivedPoints, class DerivedGradPoints>
  lagrange_basis(int degree, const Eigen::MatrixBase<DerivedPoints>& points,
                 const Eigen::MatrixBase<DerivedGradPoints>& grad_points)
      : Base(degree), mono_basis_(degree) {
    POLATORY_ASSERT(points.rows() == basis_size() ||
                    degree == 1 && points.rows() == 1 && grad_points.rows() == 1);

    Eigen::MatrixXd p = mono_basis_.evaluate(points, grad_points).transpose();

    Eigen::FullPivLU<Eigen::MatrixXd> lu(p);

    if (!lu.isInvertible()) {
      throw std::domain_error("The set of points is not unisolvent.");
    }

    coeffs_ = lu.inverse();
    rcond_ = lu.rcond();
  }

  template <class Derived>
  Eigen::MatrixXd evaluate(const Eigen::MatrixBase<Derived>& points) const {
    return evaluate(points, Points(0, Dim));
  }

  template <class DerivedPoints, class DerivedGradPoints>
  Eigen::MatrixXd evaluate(const Eigen::MatrixBase<DerivedPoints>& points,
                           const Eigen::MatrixBase<DerivedGradPoints>& grad_points) const {
    auto pt = mono_basis_.evaluate(points, grad_points);

    return coeffs_.transpose() * pt;
  }

  double rcond() const { return rcond_; }

 private:
  const MonomialBasis mono_basis_;

  Eigen::MatrixXd coeffs_;
  double rcond_{};
};

}  // namespace polatory::polynomial
