#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <stdexcept>
#include <vector>

namespace polatory {
namespace rbf {
namespace reference {

template <int Dim>
class cov_spherical final : public covariance_function_base<Dim> {
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

 public:
  using Base::Base;

  explicit cov_spherical(const std::vector<double>& params) { Base::set_parameters(params); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    return r < range ? psill * (1.0 - 1.5 * r / range + 0.5 * std::pow(r / range, 3.0)) : 0.0;
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    if (r < range) {
      auto coeff = psill * 1.5 * (-1.0 / (range * r) + r / std::pow(range, 3.0));
      return coeff * diff;
    }

    return Vector::Zero();
  }

  Matrix evaluate_hessian_isotropic(const Vector& /*diff*/) const override {
    throw std::runtime_error("cov_spherical::evaluate_hessian_isotropic is not implemented");
  }
};

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
