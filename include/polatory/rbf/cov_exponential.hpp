#pragma once

#include <cmath>
#include <memory>
#include <polatory/rbf/covariance_function_base.hpp>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class cov_exponential final : public covariance_function_base<Dim> {
  using Base = covariance_function_base<Dim>;
  using matrix3d = Base::matrix3d;
  using vector3d = Base::vector3d;

 public:
  using Base::Base;

  explicit cov_exponential(const std::vector<double>& params) { Base::set_parameters(params); }

  double evaluate_isotropic(const vector3d& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    return psill * std::exp(-r / range);
  }

  vector3d evaluate_gradient_isotropic(const vector3d& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -psill * std::exp(-r / range) / (range * r);
    return coeff * diff;
  }

  matrix3d evaluate_hessian_isotropic(const vector3d& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -psill * std::exp(-r / range) / (range * r);
    return coeff *
           (matrix3d::Identity() - diff.transpose() * diff * (1.0 / (r * r) + 1.0 / (range * r)));
  }
};

}  // namespace polatory::rbf
