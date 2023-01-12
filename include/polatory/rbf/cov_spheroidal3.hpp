#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include <polatory/rbf/covariance_function_base.hpp>

namespace polatory {
namespace rbf {

class cov_spheroidal3 final : public covariance_function_base {
public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spheroidal3(const std::vector<double>& params) {
    set_parameters(params);
  }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spheroidal3>(*this);
  }

  static double evaluate_untransformed(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return r < 0.18657871684006438 * range
           ? psill * (1.0 - 2.0098755439584821 * r / range)
           : psill * 0.87346405371085535 * std::pow(1.0 + 7.1815105816931630 * std::pow(r / range, 2.0), -1.5);
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(
    double *gradx, double *grady, double *gradz,
    double x, double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = r < 0.18657871684006438 * range
             ? -psill * 2.0098755439584821 / (range * r)
             : -psill * 18.818374033359339 * std::pow(1.0 + 7.1815105816931630 * std::pow(r / range, 2.0), -2.5) / (range * range);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }
};

}  // namespace rbf
}  // namespace polatory
