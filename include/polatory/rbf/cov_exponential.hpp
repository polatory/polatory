#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class cov_exponential final : public covariance_function_base<Dim> {
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

 public:
  using Base::Base;

  explicit cov_exponential(const std::vector<double>& params) { Base::set_parameters(params); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    return psill * std::exp(-r / range);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -psill * std::exp(-r / range) / (range * r);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -psill * std::exp(-r / range) / (range * r);
    return coeff *
           (Matrix::Identity() - diff.transpose() * diff * (1.0 / (r * r) + 1.0 / (range * r)));
  }
};

}  // namespace polatory::rbf
