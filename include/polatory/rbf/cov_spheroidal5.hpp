#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory {
namespace rbf {

class cov_spheroidal5 final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal5(const std::vector<double> &params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal5>(*this);
  }

  static double evaluate_untransformed(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < 0.25801274118035729 * range
               ? psill * (1.0 - 1.6149073288415875 * r / range)
               : psill * 0.85759801680320064 *
                     std::pow(1.0 + 2.5036086535164204 * std::pow(r / range, 2.0), -2.5);
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(double *gradx, double *grady, double *gradz, double x,
                                       double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = r < 0.25801274118035729 * range
                 ? -psill * 1.6149073288415875 / (range * r)
                 : -psill * 10.735449080535068 *
                       std::pow(1.0 + 2.5036086535164204 * std::pow(r / range, 2.0), -3.5) /
                       (range * range);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }
};

}  // namespace rbf
}  // namespace polatory
