#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal9 final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal9(const std::vector<double> &params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal9>(*this);
  }

  static double evaluate_untransformed(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < 0.31622776601683793 * range
               ? psill * (1.0 - 1.4230249470757707 * r / range)
               : psill * 0.84455856903325538 * std::pow(1.0 + std::pow(r / range, 2.0), -4.5);
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(double *gradx, double *grady, double *gradz, double x,
                                       double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = r < 0.31622776601683793 * range
                 ? -psill * 1.4230249470757707 / (range * r)
                 : -psill * 7.6010271212992985 * std::pow(1.0 + std::pow(r / range, 2.0), -5.5) /
                       (range * range);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }
};

}  // namespace polatory::rbf
