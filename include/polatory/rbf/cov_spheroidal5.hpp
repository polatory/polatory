#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal5 final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal5(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal5>(*this);
  }

  double evaluate_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < 0.25801274118035729 ? psill * (1.0 - 1.6149073288415875 * rho)
                                     : psill * 0.85759801680320064 *
                                           std::pow(1.0 + 2.5036086535164204 * (rho * rho), -2.5);
  }

  vector3d evaluate_gradient_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < 0.25801274118035729 ? -psill * 1.6149073288415875 / rho
                                   : -psill * 10.735449080535068 *
                                         std::pow(1.0 + 2.5036086535164204 * (rho * rho), -3.5)) /
        (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < 0.25801274118035729 ? -psill * 1.6149073288415875 / rho
                                   : -psill * 10.735449080535068 *
                                         std::pow(1.0 + 2.5036086535164204 * (rho * rho), -3.5)) /
        (range * range);
    return coeff *
           (matrix3d::Identity() - diff.transpose() * diff *
                                       (rho < 0.25801274118035729
                                            ? 1.0 / (r * r)
                                            : 7.0 / (r * r + 0.3994234476684122 * range * range)));
  }
};

}  // namespace polatory::rbf
