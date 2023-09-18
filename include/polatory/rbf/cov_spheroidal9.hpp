#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal9 final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal9(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal9>(*this);
  }

  double evaluate_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < 0.31622776601683793
               ? psill * (1.0 - 1.4230249470757707 * rho)
               : psill * 0.84455856903325538 * std::pow(1.0 + (rho * rho), -4.5);
  }

  vector3d evaluate_gradient_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < 0.31622776601683793
                      ? -psill * 1.4230249470757707 / rho
                      : -psill * 7.6010271212992985 * std::pow(1.0 + (rho * rho), -5.5)) /
                 (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < 0.31622776601683793
                      ? -psill * 1.4230249470757707 / rho
                      : -psill * 7.6010271212992985 * std::pow(1.0 + (rho * rho), -5.5)) /
                 (range * range);
    return coeff *
           (matrix3d::Identity() -
            diff.transpose() * diff *
                (rho < 0.31622776601683793 ? 1.0 / (r * r) : 11.0 / (r * r + range * range)));
  }
};

}  // namespace polatory::rbf
