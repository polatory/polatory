#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal3 final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal3(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal3>(*this);
  }

  double evaluate_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < 0.18657871684006438
               ? psill * (1.0 - 2.0098755439584821 * rho)
               : psill * 0.87346405371085535 *
                     std::pow(1.0 + 7.1815105816931630 * std::pow(rho, 2.0), -1.5);
  }

  vector3d evaluate_gradient_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < 0.18657871684006438 ? -psill * 2.0098755439584821 / rho
                                   : -psill * 18.818374033359339 *
                                         std::pow(1.0 + 7.1815105816931630 * (rho * rho), -2.5)) /
        (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < 0.18657871684006438 ? -psill * 2.0098755439584821 / rho
                                   : -psill * 18.818374033359339 *
                                         std::pow(1.0 + 7.1815105816931630 * (rho * rho), -2.5)) /
        (range * range);
    return coeff *
           (matrix3d::Identity() - diff.transpose() * diff *
                                       (rho < 0.18657871684006438
                                            ? 1.0 / (r * r)
                                            : 5.0 / (r * r + 0.1392464703107397 * range * range)));
  }
};

}  // namespace polatory::rbf
