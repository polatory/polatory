#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal9 final : public covariance_function_base {
  static constexpr double kRho0 = 0.31622776601683794;
  static constexpr double kA = 1.4230249470757708;
  static constexpr double kB = 0.8445585690332554;
  static constexpr double kD = 7.601027121299299;

 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal9(const std::vector<double>& params) { set_parameters(params); }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho) : psill * kB * std::pow(1.0 + (rho * rho), -4.5);
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + (rho * rho), -5.5)) /
        (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + (rho * rho), -5.5)) /
        (range * range);
    return coeff * (matrix3d::Identity() -
                    diff.transpose() * diff *
                        (rho < kRho0 ? 1.0 / (r * r) : 11.0 / (r * r + range * range)));
  }
};

}  // namespace polatory::rbf
