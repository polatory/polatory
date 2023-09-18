#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal7 final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal7(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal7>(*this);
  }

  double evaluate_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < 0.29441494768436372 ? psill * (1.0 - 1.4859979204216046 * rho)
                                     : psill * 0.84948625330168548 *
                                           std::pow(1.0 + 1.4420831474268300 * (rho * rho), -3.5);
  }

  vector3d evaluate_gradient_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < 0.29441494768436372 ? -psill * 1.4859979204216046 / rho
                                   : -psill * 8.5752086689998398 *
                                         std::pow(1.0 + 1.4420831474268300 * (rho * rho), -4.5)) /
        (range * range);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_untransformed(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff =
        (rho < 0.29441494768436372 ? -psill * 1.4859979204216046 / rho
                                   : -psill * 8.5752086689998398 *
                                         std::pow(1.0 + 1.4420831474268300 * (rho * rho), -4.5)) /
        (range * range);
    return coeff *
           (matrix3d::Identity() - diff.transpose() * diff *
                                       (rho < 0.29441494768436372
                                            ? 1.0 / (r * r)
                                            : 9.0 / (r * r + 0.6934412913598931 * range * range)));
  }
};

}  // namespace polatory::rbf
