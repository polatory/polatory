#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory {
namespace rbf {
namespace reference {

class cov_gaussian final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_gaussian(const std::vector<double> &params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override { return std::make_unique<cov_gaussian>(*this); }

  static double evaluate_untransformed(double r, const double *params) {
    auto psill = params[0];
    auto range = params[1];

    return psill * std::exp(-r * r / (range * range));
  }

  double evaluate_untransformed(double r) const override {
    return evaluate_untransformed(r, parameters().data());
  }

  void evaluate_gradient_untransformed(double *gradx, double *grady, double *gradz, double x,
                                       double y, double z, double r) const override {
    auto psill = parameters()[0];
    auto range = parameters()[1];

    auto c = -2.0 * psill * std::exp(-r * r / (range * range)) / (range * range);
    *gradx = c * x;
    *grady = c * y;
    *gradz = c * z;
  }
};

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
