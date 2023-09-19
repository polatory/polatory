#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal3 final : public covariance_function_base {
  static constexpr double kRho0 = 0.18657871684006438;
  static constexpr double kA = 2.009875543958482;
  static constexpr double kB = 0.8734640537108553;
  static constexpr double kC = 7.181510581693163;
  static constexpr double kD = 18.81837403335934;
  static constexpr double kE = 0.1392464703107397;

 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal3(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal3>(*this);
  }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho)
                       : psill * kB * std::pow(1.0 + kC * (rho * rho), -1.5);
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -2.5)) /
        (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -2.5)) /
        (range * range);
    return coeff * (matrix3d::Identity() -
                    diff.transpose() * diff *
                        (rho < kRho0 ? 1.0 / (r * r) : 5.0 / (r * r + kE * range * range)));
  }
};

}  // namespace polatory::rbf
