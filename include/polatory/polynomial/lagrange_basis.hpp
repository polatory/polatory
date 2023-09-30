#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>
#include <stdexcept>

namespace polatory::polynomial {

template <int Dim>
class lagrange_basis : public polynomial_basis_base<Dim> {
  static constexpr double kRCondThreshold = 1e-10;

  using Base = polynomial_basis_base<Dim>;
  static constexpr int kDim = Dim;
  using MonomialBasis = monomial_basis<kDim>;

 public:
  using Base::basis_size;
  using Base::degree;

  template <class Derived>
  lagrange_basis(int degree, const Eigen::MatrixBase<Derived>& points)
      : Base(degree), mono_basis_(degree) {
    POLATORY_ASSERT(points.rows() == basis_size());

    Eigen::MatrixXd p = mono_basis_.evaluate(points).transpose();

    // Do not use p.fullPivLu().isInvertible() which is too robust
    // nor p.fullPivLu().rcond() which returns an inexact value.
    if (!is_invertible(p)) {
      throw std::domain_error("The set of points is not unisolvent.");
    }

    coeffs_ = p.fullPivLu().inverse();
  }

  template <class Derived>
  Eigen::MatrixXd evaluate(const Eigen::MatrixBase<Derived>& points) const {
    return evaluate(points, geometry::points3d(0, 3));
  }

  template <class DerivedPoints, class DerivedGradPoints>
  Eigen::MatrixXd evaluate(const Eigen::MatrixBase<DerivedPoints>& points,
                           const Eigen::MatrixBase<DerivedGradPoints>& grad_points) const {
    auto pt = mono_basis_.evaluate(points, grad_points);

    return coeffs_.transpose() * pt;
  }

 private:
  static bool is_invertible(const Eigen::MatrixXd& m) {
    auto svd = m.jacobiSvd();
    const auto& sigmas = svd.singularValues();
    if (sigmas(0) == 0.0) {
      return false;
    }

    auto rcond = sigmas(sigmas.rows() - 1) / sigmas(0);
    return rcond >= kRCondThreshold;
  }

  const MonomialBasis mono_basis_;

  Eigen::MatrixXd coeffs_;
};

}  // namespace polatory::polynomial
