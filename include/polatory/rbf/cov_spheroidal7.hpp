#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

class cov_spheroidal7 final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal7(const std::vector<double> &params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal7>(*this);
  }

  static double evaluate_untransformed(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < 0.29441494768436372 * range
               ? psill * (1.0 - 1.4859979204216046 * r / range)
               : psill * 0.84948625330168548 *
                     std::pow(1.0 + 1.4420831474268300 * std::pow(r / range, 2.0), -3.5);
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(double *gradx, double *grady, double *gradz, double x,
                                       double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = r < 0.29441494768436372 * range
                 ? -psill * 1.4859979204216046 / (range * r)
                 : -psill * 8.5752086689998398 *
                       std::pow(1.0 + 1.4420831474268300 * std::pow(r / range, 2.0), -4.5) /
                       (range * range);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }
};

}  // namespace polatory::rbf
