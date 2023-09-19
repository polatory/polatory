#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal7 final : public covariance_function_base {
  static constexpr double kRho0 = 0.2944149476843637;
  static constexpr double kA = 1.4859979204216045;
  static constexpr double kB = 0.8494862533016855;
  static constexpr double kC = 1.44208314742683;
  static constexpr double kD = 8.57520866899984;
  static constexpr double kE = 0.6934412913598931;

 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal7(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal7>(*this);
  }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho)
                       : psill * kB * std::pow(1.0 + kC * (rho * rho), -3.5);
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -4.5)) /
        (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + kC * (rho * rho), -4.5)) /
        (range * range);
    return coeff * (matrix3d::Identity() -
                    diff.transpose() * diff *
                        (rho < kRho0 ? 1.0 / (r * r) : 9.0 / (r * r + kE * range * range)));
  }
};

}  // namespace polatory::rbf
