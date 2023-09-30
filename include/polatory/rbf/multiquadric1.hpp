#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <polatory/rbf/rbf_base.hpp>
#include <vector>

namespace polatory::rbf {

template <int Dim>
class multiquadric1 final : public rbf_base<Dim> {
  using Base = rbf_base<Dim>;
  using Matrix = Base::Matrix;
  using Vector = Base::Vector;

 public:
  using Base::Base;

  explicit multiquadric1(const std::vector<double>& params) { Base::set_parameters(params); }

  int cpd_order() const override { return 1; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto c = Base::parameters().at(1);
    auto r = diff.norm();

    return -slope * std::hypot(r, c);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto c = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -slope / std::hypot(r, c);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto c = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -slope / std::hypot(r, c);
    return coeff * (Matrix::Identity() - diff.transpose() * diff / (r * r + c * c));
  }

  int num_parameters() const override { return 2; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0, 0.0};
    return lower_bounds;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity(),
                                                  std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

}  // namespace polatory::rbf
