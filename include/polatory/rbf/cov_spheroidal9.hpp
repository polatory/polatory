#pragma once

#include <cmath>
#include <polatory/rbf/covariance_function_base.hpp>
#include <polatory/rbf/rbf.hpp>
#include <vector>

namespace polatory::rbf {

namespace internal {

template <int Dim>
class CovSpheroidal9 final : public CovarianceFunctionBase<Dim> {
 public:
  static constexpr int kDim = Dim;
  static inline const std::string kShortName = "sp9";

 private:
  using Base = CovarianceFunctionBase<Dim>;
  using Mat = Base::Mat;
  using RbfPtr = Base::RbfPtr;
  using Vector = Base::Vector;

  static constexpr double kRho0 = 0.31622776601683794;
  static constexpr double kA = 1.4230249470757708;
  static constexpr double kB = 0.8445585690332554;
  static constexpr double kD = 7.601027121299299;

 public:
  using Base::Base;
  using Base::parameters;
  using Base::set_parameters;

  explicit CovSpheroidal9(const std::vector<double>& params) { set_parameters(params); }

  RbfPtr clone() const override { return std::make_unique<CovSpheroidal9>(*this); }

  double evaluate_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    return rho < kRho0 ? psill * (1.0 - kA * rho) : psill * kB * std::pow(1.0 + rho * rho, -4.5);
  }

  Vector evaluate_gradient_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + rho * rho, -5.5)) /
                 (range * range);
    return coeff * diff;
  }

  Mat evaluate_hessian_isotropic(const Vector& diff) const override {
    auto psill = parameters().at(0);
    auto range = parameters().at(1);
    auto r = diff.norm();
    auto rho = r / range;

    auto coeff = (rho < kRho0 ? -psill * kA / rho : -psill * kD * std::pow(1.0 + rho * rho, -5.5)) /
                 (range * range);
    return coeff *
           (Mat::Identity() - (rho < kRho0 ? 1.0 / (r * r) : 11.0 / (r * r + range * range)) *
                                  diff.transpose() * diff);
  }

  std::string short_name() const override { return kShortName; }
};

}  // namespace internal

POLATORY_DEFINE_RBF(CovSpheroidal9);

}  // namespace polatory::rbf
