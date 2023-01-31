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

class lagrange_basis : public polynomial_basis_base {
  static constexpr double kRCondThreshold = 1e-10;

 public:
  lagrange_basis(int dimension, int degree, const geometry::points3d& points)
      : polynomial_basis_base(dimension, degree), mono_basis_(dimension, degree) {
    POLATORY_ASSERT(points.rows() == basis_size());

    Eigen::MatrixXd p = mono_basis_.evaluate(points).transpose();

    // Do not use p.fullPivLu().isInvertible() which is too robust
    // nor p.fullPivLu().rcond() which returns an inexact value.
    if (!is_invertible(p)) {
      throw std::domain_error("The set of points is not unisolvent.");
    }

    coeffs_ = p.fullPivLu().inverse();
  }

  Eigen::MatrixXd evaluate(const geometry::points3d& points) const {
    auto pt = mono_basis_.evaluate(points);

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

  const monomial_basis mono_basis_;

  Eigen::MatrixXd coeffs_;
};

}  // namespace polatory::polynomial
