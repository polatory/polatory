#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class cov_spheroidal5 final : public covariance_function_base<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

  static constexpr double kRho0 = 0.2580127411803573;
  static constexpr double kA = 1.6149073288415876;
  static constexpr double kB = 0.8575980168032007;
  static constexpr double kC = 2.5036086535164204;
  static constexpr double kD = 10.735449080535068;
  static constexpr double kE = 0.39942344766841226;

 public:
  using Base::Base;

  explicit cov_spheroidal5(const std::vector<double>& params) { Base::set_parameters(params); }

  RbfPtr<kDim> clone() const override { return std::make_unique<cov_spheroidal5>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho)
                       : psill * kB * std::pow(1.0 + kC * (rho * rho), -2.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -3.5)) /
        (range * range);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -3.5)) /
        (range * range);
    return coeff * (Matrix::Identity() -
                    (rho < kRho0 ? 1.0 / (r * r) : 7.0 / (r * r + kE * range * range)) *
                        diff.transpose() * diff);
  }
};

}  // namespace polatory::rbf
