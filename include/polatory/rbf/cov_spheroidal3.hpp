#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class cov_spheroidal3 final : public covariance_function_base<Dim> {
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

  static constexpr double kRho0 = 0.18657871684006438;
  static constexpr double kA = 2.009875543958482;
  static constexpr double kB = 0.8734640537108553;
  static constexpr double kC = 7.181510581693163;
  static constexpr double kD = 18.81837403335934;
  static constexpr double kE = 0.1392464703107397;

 public:
  using Base::Base;

  explicit cov_spheroidal3(const std::vector<double>& params) { Base::set_parameters(params); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho)
                       : psill * kB * std::pow(1.0 + kC * (rho * rho), -1.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -2.5)) /
        (range * range);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -2.5)) /
        (range * range);
    return coeff * (Matrix::Identity() -
                    (rho < kRho0 ? 1.0 / (r * r) : 5.0 / (r * r + kE * range * range)) *
                        diff.transpose() * diff);
  }
};

}  // namespace polatory::rbf
