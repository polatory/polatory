#pragma once

#include <cmath>
#include <limits>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, int k>
class inverse_multiquadric final : public rbf_base<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = rbf_base<Dim>;
  using Matrix = Base::Matrix;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

  static_assert(k > 0 && k % 2 == 1, "k must be a positive odd integer.");

 public:
  using Base::Base;

  explicit inverse_multiquadric(const std::vector<double>& params) { Base::set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<inverse_multiquadric>(*this); }

  int cpd_order() const override { return 0; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto c = Base::parameters().at(1);
    auto r = diff.norm();

    return slope / std::pow(std::hypot(r, c), k);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto c = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -k * slope / std::pow(std::hypot(r, c), k + 2);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = Base::parameters().at(0);
    auto c = Base::parameters().at(1);
    auto r = diff.norm();

    auto coeff = -k * slope / std::pow(std::hypot(r, c), k + 2);
    return coeff * (Matrix::Identity() - (k + 2) / (r * r + c * c) * diff.transpose() * diff);
  }

  int num_parameters() const override { return 2; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0, 0.0};
    return lower_bounds;
  }

  const std::vector<std::string>& parameter_names() const override {
    static const std::vector<std::string> names{"scale", "c"};
    return names;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity(),
                                                  std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

template <int Dim>
using inverse_multiquadric1 = inverse_multiquadric<Dim, 1>;

}  // namespace internal

DEFINE_RBF(inverse_multiquadric1);

}  // namespace polatory::rbf
