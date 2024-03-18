#pragma once

#include <limits>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <string>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim, int k>
class polyharmonic_odd final : public rbf_base<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = rbf_base<Dim>;
  using Matrix = Base::Matrix;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

  static_assert(k > 0 && k % 2 == 1, "k must be a positive odd integer.");

  static constexpr double kSign = ((k + 1) / 2) % 2 == 0 ? 1.0 : -1.0;

 public:
  using Base::Base;
  using Base::parameters;

  explicit polyharmonic_odd(const std::vector<double>& params) { Base::set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<polyharmonic_odd>(*this); }

  int cpd_order() const override { return (k + 1) / 2; }

  double evaluate_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    return kSign * slope * std::pow(r, k);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Vector::Zero();
    }

    auto coeff = kSign * k * slope * std::pow(r, k - 2);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto slope = parameters().at(0);
    auto r = diff.norm();

    if (r == 0.0) {
      return Matrix::Zero();
    }

    auto coeff = kSign * k * slope * std::pow(r, k - 2);
    return coeff * (Matrix::Identity() + (k - 2) / (r * r) * diff.transpose() * diff);
  }

  int num_parameters() const override { return 1; }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{0.0};
    return lower_bounds;
  }

  const std::vector<std::string>& parameter_names() const override {
    static const std::vector<std::string> names{"scale"};
    return names;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{std::numeric_limits<double>::infinity()};
    return upper_bounds;
  }
};

template <int Dim>
using biharmonic3d = polyharmonic_odd<Dim, 1>;

template <int Dim>
using triharmonic3d = polyharmonic_odd<Dim, 3>;

}  // namespace internal

DEFINE_RBF(biharmonic3d);
DEFINE_RBF(triharmonic3d);

}  // namespace polatory::rbf
