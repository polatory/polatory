#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class CovGeneralizedCauchy5 final : public CovarianceFunctionBase<Dim> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "gc5";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = Base::Mat;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

  static constexpr double kA = 2.4822022531844965;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovGeneralizedCauchy5(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovGeneralizedCauchy5>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return psill * std::pow(1.0 + kA * rho * rho, -2.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -kA * 5.0 * psill * std::pow(1.0 + kA * rho * rho, -3.5) / (range * range);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -kA * 5.0 * psill * std::pow(1.0 + kA * rho * rho, -3.5) / (range * range);
    return coeff *
           (Mat::Identity() - kA * 7.0 / (kA * r * r + range * range) * diff.transpose() * diff);
  }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(CovGeneralizedCauchy5);

}  // namespace polatory::rbf
