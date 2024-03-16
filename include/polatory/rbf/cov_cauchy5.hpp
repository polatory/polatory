#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class cov_cauchy5 final : public covariance_function_base<Dim> {
 public:
  static constexpr int kDim = Dim;

 private:
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

  static constexpr double kA = 2.4822022531844965;

 public:
  using Base::Base;

  explicit cov_cauchy5(const std::vector<double>& params) { Base::set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<cov_cauchy5>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return psill * std::pow(1.0 + kA * rho * rho, -2.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -kA * 5.0 * psill * std::pow(1.0 + kA * rho * rho, -3.5) / (range * range);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = Base::parameters().at(0);
    auto range = Base::parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -kA * 5.0 * psill * std::pow(1.0 + kA * rho * rho, -3.5) / (range * range);
    return coeff *
           (Matrix::Identity() - kA * 7.0 / (kA * r * r + range * range) * diff.transpose() * diff);
  }
};

}  // namespace internal

DEFINE_RBF(cov_cauchy5);

}  // namespace polatory::rbf
