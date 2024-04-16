#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class cov_generalized_cauchy3 final : public covariance_function_base<Dim> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "gc3";

 private:
  using Base = covariance_function_base<Dim>;
  using Matrix = Base::Matrix;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

  static constexpr double kA = 7.0;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit cov_generalized_cauchy3(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<cov_generalized_cauchy3>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return psill * std::pow(1.0 + kA * rho * rho, -1.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -kA * 3.0 * psill * std::pow(1.0 + kA * rho * rho, -2.5) / (range * range);
    return coeff * diff;
  }

  Matrix evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -kA * 3.0 * psill * std::pow(1.0 + kA * rho * rho, -2.5) / (range * range);
    return coeff *
           (Matrix::Identity() - kA * 5.0 / (kA * r * r + range * range) * diff.transpose() * diff);
  }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(cov_generalized_cauchy3);

}  // namespace polatory::rbf
