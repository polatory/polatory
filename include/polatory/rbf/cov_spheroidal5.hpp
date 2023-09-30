#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal5 final : public covariance_function_base {
  static constexpr double kRho0 = 0.2580127411803573;
  static constexpr double kA = 1.6149073288415876;
  static constexpr double kB = 0.8575980168032007;
  static constexpr double kC = 2.5036086535164204;
  static constexpr double kD = 10.735449080535068;
  static constexpr double kE = 0.39942344766841226;

 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal5(const std::vector<double>& params) { set_parameters(params); }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho)
                       : psill * kB * std::pow(1.0 + kC * (rho * rho), -2.5);
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -3.5)) /
        (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -3.5)) /
        (range * range);
    return coeff * (matrix3d::Identity() -
                    diff.transpose() * diff *
                        (rho < kRho0 ? 1.0 / (r * r) : 7.0 / (r * r + kE * range * range)));
  }
};

}  // namespace polatory::rbf
