#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <polatory/rbf/rbf_base.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class biharmonic2d final : public rbf_base<Dim> {
  using Base = rbf_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

 public:
  using Base::Base;

  explicit biharmonic2d(const std::vector<double>& params) { Base::set_parameters(params); }

  int cpd_order() const override { return 2; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return 0.0;
    }

    return slope * r * r * std::log(r);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Vector::Zero();
    }

    auto coeff = slope * (1.0 + 2.0 * std::log(r));
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& /*diff*/) const override {
    throw std::runtime_error("biharmonic2d::evaluate_hessian_isotropic is not implemented");
  }

  int num_parameters() const override { return 1; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0};
    return lower_bounds;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

}  // namespace polatory::rbf
