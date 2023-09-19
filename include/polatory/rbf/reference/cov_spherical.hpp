#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <stdexcept>
#include <vector>

namespace polatory {
namespace rbf {
namespace reference {

class cov_spherical final : public covariance_function_base {
 public:
  using covariance_function_base::covariance_function_base;

  explicit cov_spherical(const std::vector<double>& params) { set_parameters(params); }

  std::unique_ptr<rbf_base> clone() const override {
    return std::make_unique<cov_spherical>(*this);
  }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();

    return r < range ? psill * (1.0 - 1.5 * r / range + 0.5 * std::pow(r / range, 3.0)) : 0.0;
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();

    if (r < range) {
      auto coeff = psill * 1.5 * (-1.0 / (range * r) + r / std::pow(range, 3.0));
      return coeff * diff;
    }

    return vector3d::Zero();
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& /*diff*/) const override {
    throw std::runtime_error("cov_spherical::evaluate_hessian_isotropic is not implemented");
  }
};

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
