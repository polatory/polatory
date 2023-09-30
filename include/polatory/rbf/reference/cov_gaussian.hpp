#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory {
namespace rbf {
namespace reference {

template <int Dim>
class cov_gaussian final : public covariance_function_base<Dim> {
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

 public:
  using Base::Base;

  explicit cov_gaussian(const std::vector<double>& params) { Base::set_parameters(params); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    return psill * std::exp(-r * r / (range * range));
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -2.0 * psill * std::exp(-r * r / (range * range)) / (range * range);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -2.0 * psill * std::exp(-r * r / (range * range)) / (range * range);
    return coeff * (Matrix::Identity() - 2.0 * diff.transpose() * diff / (range * range));
  }
};

}  // namespace reference
}  // namespace rbf
}  // namespace polatory
