#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class cov_spheroidal9 final : public covariance_function_base<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

  static constexpr double kRho0 = 0.31622776601683794;
  static constexpr double kA = 1.4230249470757708;
  static constexpr double kB = 0.8445585690332554;
  static constexpr double kD = 7.601027121299299;

 public:
  using Base::Base;

  explicit cov_spheroidal9(const std::vector<double>& params) { Base::set_parameters(params); }

  RbfPtr<kDim> clone() const override { return std::make_unique<cov_spheroidal9>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho) : psill * kB * std::pow(1.0 + rho * rho, -4.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + rho * rho, -5.5)) /
                 (range * range);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + rho * rho, -5.5)) /
                 (range * range);
    return coeff *
           (Matrix::Identity() - (rho < kRho0 ? 1.0 / (r * r) : 11.0 / (r * r + range * range)) *
                                     diff.transpose() * diff);
  }
};

}  // namespace internal

DEFINE_RBF(cov_spheroidal9);

}  // namespace polatory::rbf
