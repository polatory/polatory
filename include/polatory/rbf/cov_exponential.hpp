#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class CovExponential final : public CovarianceFunctionBase<Dim> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "exp";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = typename Base::Mat;
  using RbfPtr = typename Base::RbfPtr;
  using Vector = typename Base::Vector;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovExponential(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovExponential>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return psill * std::exp(-3.0 * rho);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -3.0 * psill * std::exp(-3.0 * rho) / (range * r);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = -3.0 * psill * std::exp(-3.0 * rho) / (range * r);
    return coeff *
           (Mat::Identity() - (1.0 / (r * r) + 3.0 / (range * r)) * diff.transpose() * diff);
  }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(CovExponential);

}  // namespace polatory::rbf
